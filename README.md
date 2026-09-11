# Progetto-AI-VacuumCleaner
    Da 0:
    1 - scarica dataset emnist
    2 - prepara dataset emnist
    3 - genera dataset digitale
    4 - combina dataset
    5 - train
    6 - evaluate
    7 - predict_table che sfrutta : smartvacuum, predict
# Classificatore
    Classificatore

    Il classificatore riconosce le seguenti lettere maiuscole:

    C, D, F, S, V, X

    Il dataset utilizzato comprende:

    lettere manoscritte, tramite EMNIST ByClass;
    lettere digitali generate automaticamente tramite diversi font.
    Installazione

    Dopo aver clonato il repository, installare le dipendenze:

    python3 -m pip install -r requisiti.txt

    Preparazione dei dataset
    1. Scaricare EMNIST:
    python3 -m src.classificatore.scarica_dataset_emnist
    2. Preparare il dataset EMNIST:
    python3 -m src.classificatore.prepara_dataset_emnist
    3. Generare il dataset digitale:
    python3 -m src.classificatore.genera_dataset_digitale

    I dataset vengono salvati automaticamente nella cartella data/.

    La cartella data/ non viene caricata su GitHub, quindi ogni sviluppatore deve generare localmente i propri dataset.

    Verifica

    Per verificare visivamente le immagini EMNIST:

    python3 tests/test_visualizzazione_emnist.py

    Per verificare le immagini digitali:

    python3 tests/test_visualizzazione_dataset_digitale.py


    Nota

    I dataset non sono versionati su GitHub perché possono essere ricreati automaticamente tramite gli script presenti nel progetto.

# 