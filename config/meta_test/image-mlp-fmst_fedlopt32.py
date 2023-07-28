_base_ = ["./meta_test_base.py"]


local_learning_rate = 0.5
hidden_size = 32
task = "small-image-mlp-fmst"
optimizer = "fedlopt"
test_checkpoint = "models/small-image-mlp/fedlopt32_small-image-mlp-fmst_K8_H4_0.5.pickle"