import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, RelaxedBernoulli
from src.models.layers.base import MLP
from src.utils import numerical_stability_check
from src.utils import SCBMPercentileStrategy

class SCBM(nn.Module):
    """
    Adaptation of the Stochastic Concept Bottleneck Model (SCBM):
        - paper: https://arxiv.org/pdf/2406.19272v1
        - code: https://github.com/mvandenhi/SCBM/blob/main/models/models.py#L30
    """
    def __init__(self, 
                 input_size, 
                 hidden_size,
                 concept_hidden_size,
                 output_size=2,
                 n_layers_encoder=1,
                 n_layers_concept_encoder=1,
                 n_layers_decoder=1,
                 activation='leaky_relu',
                 concept_loss_weight=0.5,
                 decoder_type='mlp',
                 cov_type= "amortized",
                 num_monte_carlo=1,
                 max_epochs=100,
                 alpha=1,
                 reg_weight=1,
                 reg_precision="l1",
                 c_info={},
                 y_info={},
                 name: str = 'SCBM'):
        super(SCBM, self).__init__()
        
        # to be stored for every model
        self.has_concepts = True
        self.is_causal = False
        self.max_epochs = max_epochs
        self.output_size = output_size
        self.name = name

        # model specific
        self.cov_type = cov_type
        self.num_monte_carlo = num_monte_carlo
        self.training_epoch = 0
        self.alpha = alpha
        self.reg_weight = reg_weight
        self.reg_precision = reg_precision
        self.concept_pred = []
        self.concept_pred_tmp = []
        self.interv_strat = SCBMPercentileStrategy()
        self.act_c = nn.Sigmoid()

        # concepts info
        self.concept_names = c_info['names']
        self.virtual_roots = [name for name in c_info['names'] if name.startswith('#virtual_')]
        self.concept_loss_weight = concept_loss_weight
        self.filtered_c_info = {
            "names": [name for name in c_info["names"] if name not in self.virtual_roots],
            "cardinality": [card for name, card in zip(c_info["names"], c_info["cardinality"]) if name not in self.virtual_roots]
        }
        # number of concepts
        self.num_concepts = len(self.filtered_c_info['names'])
        self.sum_cardinailities = sum(self.filtered_c_info['cardinality'])

        # Encoder
        self.encoder = MLP(input_size=input_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers_encoder,
                           activation=activation)

        # expected value for the eta random variable
        self.mu_mlp = MLP(input_size=hidden_size,
                         hidden_size=concept_hidden_size,
                         output_size=sum(self.filtered_c_info['cardinality']),
                         n_layers=n_layers_concept_encoder,
                         activation='linear')
        
        # Covariance matrix for the concepts
        if self.cov_type == "global":
            self.sigma_concepts = nn.Parameter(
                torch.zeros(int(self.sum_cardinailities * (self.sum_cardinailities + 1) / 2))
                ) # Predict lower triangle of concept covariance
        elif self.cov_type == "amortized":
            self.sigma_concepts = nn.Linear(
                hidden_size,
                int(self.sum_cardinailities * (self.sum_cardinailities + 1) / 2),
                bias=True,
            )
            self.sigma_concepts.weight.data *= (
                0.01  # To prevent exploding precision matrix at initialization
            )
        else:
            raise ValueError(f"Invalid covariance type {self.cov_type}")

        # Decoder
        if decoder_type == 'mlp':
            self.decoder = MLP(input_size=sum(self.filtered_c_info['cardinality']),
                               hidden_size=sum(self.filtered_c_info['cardinality'])//2,
                               output_size=output_size,
                               n_layers=n_layers_decoder,
                               activation=activation)

        elif decoder_type == 'linear':
            self.decoder = nn.Linear(sum(self.filtered_c_info['cardinality']), 
                                     output_size)
        else:
            raise ValueError(f"Decoder type {decoder_type} not supported")

    '''
    def act_c(self, x):
        # apply a softmax to each concept separately (different concepts might have different cardinalities)
        concept_list = []
        for i, name in enumerate(self.filtered_c_info['names']):
            concept_list.append(F.softmax(x[:,sum(self.filtered_c_info['cardinality'][:i]):sum(self.filtered_c_info['cardinality'][:i+1]),:], dim=1))
        return torch.cat(concept_list, dim=1)
    '''

    def forward(self, x, c=None, intervention_index=None):
        bsz = x.shape[0]
        intermediate = self.encoder(x)

        # Compute the expected value for the concepts
        c_mu = self.mu_mlp(intermediate)

        # Compute the covariance matrix for the concepts
        if self.cov_type == "global":
            c_sigma = self.sigma_concepts.repeat(c_mu.size(0), 1)
        elif self.cov_type == "amortized":
            c_sigma = self.sigma_concepts(intermediate)
        else:
            raise ValueError(f"Invalid covariance type {self.cov_type}")

        # Fill the lower triangle of the covariance matrix with the values and make diagonal positive
        c_triang_cov = torch.zeros(
            (c_sigma.shape[0], self.sum_cardinailities, self.sum_cardinailities),
            device=c_sigma.device,
        )
        rows, cols = torch.tril_indices(
            row=self.sum_cardinailities, col=self.sum_cardinailities, offset=0
        )
        diag_idx = rows == cols
        c_triang_cov[:, rows, cols] = c_sigma
        c_triang_cov[:, range(self.sum_cardinailities), range(self.sum_cardinailities)] = (
            F.softplus(c_sigma[:, diag_idx]) + 1e-6
        )

        # Sample the eta random variable
        c_dist = MultivariateNormal(c_mu, scale_tril=c_triang_cov)
        c_mcmc_logit = c_dist.rsample([self.num_monte_carlo]).movedim(
            0, -1
        )  # [batch_size,num_concepts,mcmc_size]

        c_mcmc_prob = self.act_c(c_mcmc_logit)

        # If we are in test mode and we have an intervention index, we intervene on the concepts
        # and update the relative probabilities
        if not self.training and (intervention_index is not None and not bool(intervention_index.sum()==0)) and self.training_epoch > 0:
            c_mu, c_triang_cov, c_mcmc_logit, c_mcmc_prob = self.maybe_intervene_scbm(c_mu, c_triang_cov, c, intervention_index)


        # For all MCMC samples simultaneously sample from Bernoulli
        if not self.training:
            # No backpropagation necessary
            c_mcmc = torch.bernoulli(c_mcmc_prob)
        else:
            # Backpropagation necessary
            curr_temp = self.compute_temperature(self.training_epoch, device=c_mcmc_prob.device)
            dist = RelaxedBernoulli(temperature=curr_temp, probs=c_mcmc_prob)

            # Sample from relaxed Bernoulli
            mcmc_relaxed = dist.rsample()
            # Straight-Through Gumbel Softmax
            mcmc_hard = (mcmc_relaxed > 0.5) * 1
            c_mcmc = mcmc_hard - mcmc_relaxed.detach() + mcmc_relaxed

        # store the concept predictions
        self.concept_pred_tmp.append(c_mcmc.mean(-1))

        c_mcmc_prob_dict = {}
        c_mcmc_sampled = {}
        for i, name in enumerate(self.filtered_c_info['names']):
            c_mcmc_prob_dict[name] = c_mcmc_prob[:,sum(self.filtered_c_info['cardinality'][:i]):sum(self.filtered_c_info['cardinality'][:i+1])]
            c_mcmc_sampled[name] = c_mcmc[:,sum(self.filtered_c_info['cardinality'][:i]):sum(self.filtered_c_info['cardinality'][:i+1]),:]

        # MCMC loop for predicting label
        y_pred_probs_i = torch.zeros(bsz, self.output_size, device=x.device)  
        for i in range(self.num_monte_carlo):
            c_i = torch.cat([c_mcmc_sampled[name][:,:,i] for name in self.filtered_c_info['names']], dim=1)
            y_pred_logits_i = self.decoder(c_i)
            y_pred_probs_i += torch.softmax(y_pred_logits_i, dim=1)
        y_pred_probs = y_pred_probs_i / self.num_monte_carlo

        # In order to preserve the original output shape of the forward method (y_pres, c_preds),
        # we store the covariance matrix as an attribute of the model. Ultimately using this attribute for the loss computation.
        self.c_triang_cov = c_triang_cov
        self.c_mu = c_mu

        return y_pred_probs, c_mcmc_prob_dict

    def compute_percentiles(self):
        concept_pred_percentiles = torch.quantile(
            self.concept_pred, q=torch.tensor([0.05, 0.95]).to(self.concept_pred.device), dim=0
        )
        return concept_pred_percentiles

    def _compute_intervened_perc(self, c_true, c_mask, c_mu):
        self.concept_pred_percentiles = self.compute_percentiles()

        c_true_pred_perc = torch.where(
            c_true == 1,
            self.concept_pred_percentiles[1, :],
            self.concept_pred_percentiles[0, :],
        )
        return c_true_pred_perc * c_mask

    def compute_intervened_logits(self, c_mu, c_cov, c_true, c_mask):
        c_intervened_probs = self._compute_intervened_perc(c_true, c_mask, c_mu)
        c_intervened_logits = torch.logit(c_intervened_probs, eps=1e-6)
        return c_intervened_logits

    def compute_temperature(self, epoch, device):
        # Linearly decrease the temperature from 1 (soft sampling) to 1e-6 (hard sampling)
        final_temp = torch.tensor([1e-6], device=device)
        init_temp = torch.tensor([1], device=device)
        rate = (init_temp - final_temp) / float(self.max_epochs)
        curr_temp = torch.clamp(init_temp - rate * epoch, min=final_temp.item())
        #self.curr_temp = curr_temp
        return curr_temp    

    def maybe_intervene_scbm(self, c_mu, c_cov, c_true, c_mask):
        # reshape c_true such that, according to the cardinality of self.filtered_c_info['cardinality'],
        # a one hot encoding is applied to the concept values 
        c_true = [F.one_hot(c_true[:,i].long(), num_classes=x) for i, x in enumerate(self.filtered_c_info['cardinality'])]
        c_mask = [c_mask[:,i].unsqueeze(-1).repeat(1, x) for i, x in enumerate(self.filtered_c_info['cardinality'])]
        # concatenate over dimension 1 to get the final tensor
        c_true = torch.cat(c_true, dim=1)
        c_mask = torch.cat(c_mask, dim=1)
        num_intervened = int(c_mask.sum(1)[0].item())

        # Compute the covariance matrix for the concepts
        c_cov = torch.matmul(
            c_cov,
            torch.transpose(c_cov, dim0=1, dim1=2),
        )

        device = c_mask.device

        # Compute logits of intervened-on concepts
        c_intervened_logits = self.interv_strat.compute_intervened_logits(
            c_mu, c_cov, c_true, c_mask
        )

        ## Compute conditional normal distribution sample-wise
        # Permute covariance s.t. intervened-on concepts are a block at start
        indices = torch.argsort(c_mask, dim=1, descending=True, stable=True)
        perm_cov = c_cov.gather(
            1, indices.unsqueeze(2).expand(-1, -1, c_cov.size(2))
        )
        perm_cov = perm_cov.gather(
            2, indices.unsqueeze(1).expand(-1, c_cov.size(1), -1)
        )
        perm_mu = c_mu.gather(1, indices)
        perm_c_intervened_logits = c_intervened_logits.gather(1, indices)

        # Compute mu and covariance conditioned on intervened-on concepts
        # Intermediate steps
        perm_intermediate_cov = torch.matmul(
            perm_cov[:, num_intervened:, :num_intervened],
            torch.inverse(perm_cov[:, :num_intervened, :num_intervened]),
        )
        perm_intermediate_mu = (
            perm_c_intervened_logits[:, :num_intervened]
            - perm_mu[:, :num_intervened]
        )
        # Mu and Cov
        perm_interv_mu = perm_mu[:, num_intervened:] + torch.matmul(
            perm_intermediate_cov, perm_intermediate_mu.unsqueeze(-1)
        ).squeeze(-1)
        perm_interv_cov = perm_cov[
            :, num_intervened:, num_intervened:
        ] - torch.matmul(
            perm_intermediate_cov, perm_cov[:, :num_intervened, num_intervened:]
        )

        # Adjust for floating point errors in the covariance computation to keep it symmetric
        perm_interv_cov = numerical_stability_check(
            perm_interv_cov, device=device
        )  # Uncomment if Normal throws an error. Takes some time so maybe code it more smartly

        # Sample from conditional normal
        perm_dist = MultivariateNormal(
            perm_interv_mu, covariance_matrix=perm_interv_cov
        )
        perm_mcmc_logits = (
            perm_dist.rsample([self.num_monte_carlo])
            .movedim(0, -1)
            .to(torch.float32)
        )  # [bottleneck_size-num_intervened,mcmc_size]

        # Concat logits of intervened-on concepts
        perm_mcmc_logits = torch.cat(
            (
                perm_c_intervened_logits[:, :num_intervened]
                .unsqueeze(-1)
                .repeat(1, 1, self.num_monte_carlo),
                perm_mcmc_logits,
            ),
            dim=1,
        )

        # Permute back into original form and store
        indices_reversed = torch.argsort(indices)
        mcmc_logits = perm_mcmc_logits.gather(
            1,
            indices_reversed.unsqueeze(2).expand(-1, -1, perm_mcmc_logits.size(2)),
        )

        # Return conditional mu&cov
        assert (
            torch.argsort(indices[:, num_intervened:])
            == torch.arange(len(perm_interv_mu[0][:]), device=device)
        ).all()  # Check that non-intervened concepts weren't permuted s.t. no permutation of interv_mu is needed
        interv_mu = perm_interv_mu
        interv_cov = perm_interv_cov

        assert (
            (mcmc_logits.isnan()).any()
            == (interv_mu.isnan()).any()
            == (interv_cov.isnan()).any()
            == False
        )

        # Compute probabilities and set intervened-on probs to 0/1
        mcmc_probs = self.act_c(mcmc_logits)

        # Set intervened-on hard concepts to 0/1
        mcmc_probs = (c_true * c_mask).unsqueeze(2).repeat(
            1, 1, self.num_monte_carlo
        ) + mcmc_probs * (1 - c_mask).unsqueeze(2).repeat(1, 1, self.num_monte_carlo)

        return interv_mu, interv_cov, mcmc_logits, mcmc_probs

    def filter_output_for_loss(self, y_output, c_output):
        """Filter output for loss function"""
        return y_output, c_output

    def filter_output_for_metric(self, y_output, c_output):
        """Filter output for metric function"""
        return y_output, c_output

    def loss(self, target_pred_logits, target_true, concepts_mcmc_probs, concepts_true):

        concepts_true = [F.one_hot(concepts_true[:,i].long(), num_classes=x) for i, x in enumerate(self.filtered_c_info['cardinality'])]
        concepts_true = torch.cat(concepts_true, dim=1)
        concepts_true = concepts_true.unsqueeze(-1).expand(-1, -1, self.num_monte_carlo)

        concepts_mcmc_probs = torch.cat([concepts_mcmc_probs[name] for name in self.filtered_c_info['names']], dim=1)

        c_triang_cov = self.c_triang_cov
        
        bce_loss = F.binary_cross_entropy(
            concepts_mcmc_probs, concepts_true.float(), reduction="none"
        )  # [B,C,MCMC]
        intermediate_concepts_loss = -torch.sum(bce_loss, dim=1)  # [B,MCMC]
        mcmc_loss = -torch.logsumexp(
            intermediate_concepts_loss, dim=1
        )  # [B], logsumexp for numerical stability due to shift invariance
        concepts_loss = torch.mean(mcmc_loss)

        # Compute task loss term
        loss_form = torch.nn.NLLLoss(reduction="none")

        y_hat = torch.log(target_pred_logits + 1e-6)
        y = target_true.flatten().long()
        task_loss = loss_form(y_hat, y).mean()

        # Add precision loss
        if self.reg_precision == "l1":
            c_triang_inv = torch.inverse(c_triang_cov)
            prec_matrix = torch.matmul(
                torch.transpose(c_triang_inv, dim0=1, dim1=2), c_triang_inv
            )
            prec_loss = prec_matrix.abs().sum(dim=(1, 2)) - prec_matrix.diagonal(
                offset=0, dim1=1, dim2=2
            ).abs().sum(-1)
            if prec_matrix.size(1) > 1:
                prec_loss = prec_loss / (
                    prec_matrix.size(1) * (prec_matrix.size(1) - 1)
                )
            else:  # Univariate case, can happen when intervening
                prec_loss = prec_loss
            prec_loss = prec_loss.mean(-1)
        else:
            prec_loss = torch.zeros_like(concepts_loss)

        total_loss = self.concept_loss_weight * task_loss + self.alpha * concepts_loss + self.reg_weight * prec_loss
        return total_loss
