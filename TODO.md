# TODO

**Ari & Fra**
1. **IMPORTANTE**!!!!!: dobbiamo uniformare il calcolo concept accuracy. Per ora calcoliamo la concept accuracy su tutti i concetti per cbm mentre calcoliamo la concept accuracy sui concetti parenti del task per i metodi graph based. cosa facciamo? facciamo predire al modello con il grafo anche i concetti non parenti del task? **Lasciare il bottleneck invariato per CBM ed eliminare i concetti non parent del task per C2BM. Poi calcolare la concept accuracy solo sui concetti presi da C2BM.**
2. Controllare risultati sospetti di CGM e C2BM su Alarm. Far girare tutti gli esperimenti (localized e federated).
3. modificare lo spit dei grafi:
    3.1 i client non devono necessariamente avere l'annotazione sul task.
    3.2 il numero di client e numero di sottografi sono due cose distinte (già implementato in questo modo)
    3.3 i sottografi devono sempre avere almeno un nodo che non appartiene agli altri sottografi. 
4. Ci serve un numero unico che quantifichi la bontà degli interventi del federato, per poi confrontarlo con il centralized (e con il localized).
5. Manca da visualizzare la unobserved concept accuracy table. Calcolata partendo dai risultati del federated, mettendosi dal punto di vista dei vari client.
6. Aggiungere datasets:
    6.1 selezionare solo un task dal dataset multimodale.

**Dario**
...