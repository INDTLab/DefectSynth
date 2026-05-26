save_memory = False

from cldm.hack import disable_verbosity, enable_sliced_attention

disable_verbosity()

if save_memory:
    enable_sliced_attention()
