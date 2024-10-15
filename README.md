<div align="center">
  <h1 style="font-size: 2.5em; color: #3498db;">MoAI Platform</h1>
</div>

<p align="center" style="font-size: 1.2em;">
  <strong>A full-stack infrastructure software from PyTorch to GPUs for the LLM era.</strong><br/>
  <em>MoAI Platform decouples AI infrastructure from specific hardware vendors.</em><br/>
  <em>It provides virtualization of all GPU/NPUs in a cluster for higher utilization and failover.</em><br/>
  <em>The platform can scale to thousands of GPUs/NPUs with automatic parallelization and optimization.</em><br/>
  <em>Supports any multi-billion or multi-trillion parameter model for training, and serving.</em><br/>
</p>

<hr/>

<p align="center" style="font-size: 1.1em; color: #2ecc71;">
  <strong>🚀 Designed to unlock the full potential of your AI infrastructure!</strong>
</p>

![overview_01](https://github.com/user-attachments/assets/a1d7b9b5-83f6-4844-8f16-fb6a288f54b3)



## QuickStart

The **ambre-models** repository is designed to work with a cluster where the MoAI Platform is installed. To test these scripts, please contact us.

### Training

Run the training script to fine-tune the model. For example, to fine-tune the `internlm2_5-20b-chat` model:

```
bash finetuning_codes/scripts/train_internlm.py
```
### Inference

The MoAI Platform also supports deploying inference servers for your model.

1. Run the script to deploy the model:

    ```
    bash inference_codes/scripts/change_model.sh
    ```

2. Select the model number:

    ```
    Checking agent server status...
    Agent server is normal
    
    ┌───── Current Server Info ────┐
    │ Model : internlm2_5-20b-chat │
    │ LoRA : False                 │
    │ Checkpoint :                 │
    │ Server Status : NORMAL       │
    └──────────────────────────────┘

    ========== Supported Model List ==========
     1. Meta-Llama-3-70B-Instruct
     2. internlm2_5-20b-chat
    ==========================================
    
    Select Model Number [1-2/q/Q]:
    ```

3. Check the server status:

    ```bash
    bash inference_codes/scripts/check_server.sh
    ```

    Example output:

    ```
    2024-10-15 15:34:28.736 | INFO | __main__:check_server:38 - Checking agent server status...
    2024-10-15 15:34:28.754 | INFO | __main__:check_server:41 - Agent server is normal
    
    ┌───── Current Server Info ────┐
    │ Model : internlm2_5-20b-chat │
    │ LoRA : False                 │
    │ Checkpoint :                 │
    │ Server Status : NORMAL       │
    └──────────────────────────────┘
    ```

4. Chat with your model locally or build a chat platform using the API:

    ```bash
    bash inference_codes/scripts/chat.sh
    ```

    Example chat:

    ```
    [INFO] Type 'quit' to exit
    Prompt: hello
    ================================================================================
    Assistant:
    Hello! How can I assist you today?
    ```

## Supported Models

This repository supports any multi-billion or multi-trillion parameter models for training and serving. 

### Currently Supported Models:

- **Meta-Llama-3-70B-Instruct**
- **InternLM 2.5-20B Chat**

### Future Models:

Additional models will be added in future updates. Stay tuned for more!
## Learn More

| **Section**       | **Description**                                 |
|-------------------|-------------------------------------------------|
| **[Portal](https://moreh.io/)**        | Overview of technologies and company            |
| **[Documentation](https://docs.moreh.io/)** | Detailed explanation of technology and tutorial |
| **[ModelHub](https://model-hub.moreh.io/)**     | Chatbot using the MoAI Platform solution        |


---
