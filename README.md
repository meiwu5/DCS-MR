## DCS-MR
![model](https://github.com/user-attachments/assets/8bd068bd-9e94-4b40-a267-360f2c0ed69d)
### Run Code
The dataset is obtained from:
Make sure to place it in the data directory.
'''
conda create -n DCS-MR pythom==3.12
conda activate DCS-MR
pip install -r requirements.txt
sh run.sh
'''
### Important Parameters in `config`

- **`data_process`:**  
  - `1`: Generate `FastData`.  
  - `0`: Use existing data directly without generating `FastData`.

- **`auto_test`:**  
  - `1`: Perform training.  
  - `0`: Skip training and directly use the parameter file for inference.

- **`auto_lr`:**  
  - `1`: Enable automatic learning rate adjustment.

- **`increase`:**  
  - `true`: Include nodes with increasing $v_\tau/v_{\tau-1}$ values.

- **`num_hops`:**  
  - `2`: Indicates that the sampler selects a two-hop neighborhood of nodes.

- **`replay`:**  
  - `true`: Use DC Sampler to replay nodes.

- **`replay_strategy`:**  
  - `"feature"`: Use the Target branch for feature extraction of nodes.

- **`replay_ratio`:**  
  - Represents the proportion of replayed nodes relative to the total number of current nodes.

- **`is_TMRB`:**  
  - `true`: Enable TMRB (Temporal Memory Replay Buffer).

- **`is_update`:**  
  - `true`: Update features within TMRB.

- **`select_k`:**  
  - `true`: Select nodes using the Top-K difference in features.  
  - `false`: Select nodes randomly.

