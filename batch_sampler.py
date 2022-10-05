import numpy as np
def main():
    batch_size = 1024
    for idx, i in enumerate(range(10000, 100000)):
        batch_steps = np.rint(get_num_of_batches(i))
        for i in range(batch_steps):
            batch_sources = np.random.choice([0,1], size=(batch_size), p=[idx/1e6, 1-idx/1e6]) # 0 - real, 1 - pretrain
            print(f"{idx}: {batch_steps}")
    pass

def get_num_of_batches(x):
    iter_num = 548076/(x+96152)

    return iter_num

if __name__ == "__main__":
    main()
