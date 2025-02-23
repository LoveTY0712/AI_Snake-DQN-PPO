class CONFIG:
    explore = 4000000
    learning_rate=0.001
    gamma=0.9
    replace_target_iter=2000
    buffer_size=20000
    batch_size=256
    initial_epsilon=0.6
    final_epsilon=0.001
    use_gpu=True
    save_model_frequency=10000
    experience_requirement=1000
    save_model=True
    log_info=False