# Fine-Tuning de Phi-3 Mini con LoRA en Mac M5

Este proyecto demuestra cómo realizar fine-tuning de un modelo de lenguaje grande (LLM) usando técnicas avanzadas de PEFT (Parameter-Efficient Fine-Tuning) en una **MacBook Pro M5**, adaptando código originalmente diseñado para CUDA/GPU.

## Objetivo del Modelo

El modelo base es **Phi-3-mini-4k-instruct** (3.8B parámetros) de Microsoft, un LLM compacto y eficiente. Lo ajustamos para que hable como **Yoda** de Star Wars, usando el dataset `dvgodoy/yoda_sentences` que contiene oraciones en inglés normal y su traducción al estilo Yoda.

## Técnicas Utilizadas

### 1. **LoRA (Low-Rank Adaptation)**
- Técnica de PEFT que reduce drásticamente los parámetros entrenables
- Solo entrenamos ~2% de los parámetros del modelo (~76M de 3.8B)
- Configuración:
  - `r=8`: Rango de la descomposición de matrices
  - `lora_alpha=16`: Factor de escalado (2*r)
  - `target_modules`: `['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj']`

### 2. **SFT (Supervised Fine-Tuning)**
- Entrenamiento supervisado usando el `SFTTrainer` de TRL
- Dataset formateado en formato de chat (user/assistant)
- 10 épocas de entrenamiento
- Batch size adaptativo (4 inicial, con auto-ajuste)

### 3. **Gradient Checkpointing**
- Reduce uso de memoria durante el entrenamiento
- Esencial para entrenar en hardware limitado

## Adaptaciones para Mac M5 (Apple Silicon)

El código original estaba diseñado para GPUs NVIDIA con CUDA. Estos son los cambios realizados:

### Cambios Principales

| Aspecto | Original (CUDA) | Adaptado (Mac M5) |
|---------|----------------|-------------------|
| **Device** | `cuda:0` | `mps` (Metal Performance Shaders) |
| **Cuantización** | 4-bit con BitsAndBytes | Sin cuantización (float16) |
| **Optimizador** | `paged_adamw_8bit` | `adamw_torch` |
| **Batch Size** | 16 | 4 (con auto-ajuste) |
| **Precisión** | bfloat16 | float16 |

### Código Adaptado

#### 1. Detección de Device
```python
device_map = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(
    repo_id, 
    device_map=device_map,
    quantization_config=None,  # Sin cuantización en Mac
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32
)
```

#### 2. Configuración del Trainer
```python
sft = SFTConfig(
    gradient_checkpointing=True,
    per_device_train_batch_size=4,  # Reducido para Mac
    optim='adamw_torch',  # Compatible con Mac
    # ... otros parámetros
)
```

#### 3. LoRA sin Cuantización Previa
```python
# No usamos prepare_model_for_kbit_training() en Mac
config = LoraConfig(
    r=8,
    lora_alpha=16,
    bias='none',
    lora_dropout=0.05,
    task_type='CAUSAL_LM',
    target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj']
)
model = get_peft_model(model, config)
```

## 📦 Instalación

### 1. Configurar ambiente virtual con Python 3.13
```bash
# Instalar Python 3.13 (si no lo tienes)
brew install python@3.13

# Crear ambiente virtual
/opt/homebrew/bin/python3.13 -m venv .venv

# Activar ambiente
source .venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Dependencias principales
- `transformers==4.46.2`: Framework de Hugging Face
- `peft==0.13.2`: Métodos de fine-tuning eficiente
- `trl==0.12.1`: Trainer especializado para LLMs
- `torch`: PyTorch con soporte MPS
- `datasets==3.1.0`: Manejo de datasets
- `accelerate==1.1.1`: Entrenamiento distribuido

## Uso

### Fine-Tuning
Ejecuta todas las celdas del notebook en orden:
1. Importar librerías
2. Cargar modelo con MPS
3. Configurar LoRA
4. Preparar dataset
5. Entrenar modelo
6. Guardar adaptadores

### Inferencia
```python
sentence = 'I am a software developer and AI enthusiast.'
prompt = encode_prompt(tokenizer, sentence)
output = inference(model, tokenizer, prompt)
print(output)
```

El modelo responderá en estilo Yoda: "A software developer and AI enthusiast, I am."

## Resultados

- **Parámetros entrenables**: ~76M (~2% del total)
- **Uso de memoria**: ~7.5GB (sin cuantización)
- **Tiempo de entrenamiento**: Variable según hardware
- **Device utilizado**: MPS (Metal Performance Shaders)

## Estructura del Proyecto

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── Estrategias Avanzadas de Fine Tunning (PEFT, SFT, LoRa).ipynb
├── logs/                           # Logs de entrenamiento
├── phi3-mini-yoda-adapter/         # Modelo entrenado
└── local-phi3-mini-yoda-adapter/   # Backup local
```

## Subir a Hugging Face Hub

```python
from huggingface_hub import login
import os

token = os.getenv("HF_TOKEN")
login(token=token)
trainer.push_to_hub()
```

## Consideraciones

### Ventajas de Mac M5
- MPS (Metal) ofrece buena aceleración
- Memoria unificada eficiente
- Consumo energético bajo
- Sin necesidad de GPU externa

### Limitaciones
- Sin cuantización de 4-bits (requiere CUDA)
- Batch sizes más pequeños que en GPU
- Algunas operaciones más lentas que CUDA
- Soporte limitado de bitsandbytes

### Recomendaciones
- Usar `gradient_checkpointing=True` siempre
- Dejar `auto_find_batch_size=True` para evitar OOM
- Monitorear uso de memoria con Activity Monitor
- Considerar reducir `max_seq_length` si hay problemas de memoria

## UDA vs MPS: Diferencias Clave

| Característica | CUDA (GPU NVIDIA) | MPS (Apple Silicon) |
|----------------|-------------------|---------------------|
| Cuantización 4-bit | Soportada | No soportada |
| Cuantización 8-bit | Soportada | Limitada |
| Velocidad | Más rápida | Buena |
| Optimizadores especializados | Muchos | Limitados |
| Mem. Unificada | No | Sí |

## Referencias

- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

## Créditos

- **Modelo base**: Microsoft Phi-3-mini-4k-instruct
- **Dataset**: dvgodoy/yoda_sentences
- **Framworks**: Hugging Face Transformers, PEFT, TRL
- **Adaptación para Mac**: Este proyecto

---

**Nota**: Este proyecto es educativo y demuestra cómo adaptar código de fine-tuning diseñado para CUDA/GPU para que funcione en hardware Apple Silicon, aprovechando MPS para aceleración.
