class Constant:
    class MODEL:
        seq_type="lstm"
        n_layers=2
        nclasses=1
        attention=True
        ngrams=5
        embed_size=128

    class TRAINING:
        patience=5
        batch_size=256
        val_size=0.1
        l_rate=0.001
        nepochs=30

c = Constant()