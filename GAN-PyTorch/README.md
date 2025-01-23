Entrenar el modelo:
```bash
python train.py --model dcgan
python train.py --model wgan
```
Por defecto: dcgan

Generar imágenes: 
```bash
python generate.py --model dcgan --load_path ./model_checkpoint.pth --num_output 128
```
Por defecto:
- model dcgan
- load_path generated_images/model_ChestCT.pth 
- num_output 64

Evaluar modelo 
```bash
python eval_model.py --load_path ..\..\model\model_wgan\model_epoch_990.pth
```
Por defecto: generated_images/model_ChestCT.pth 

Generar gráfico de pérdidas del discriminador y generador

```bash
python3 graphLogs.py --csv training_log_wgan.csv
```