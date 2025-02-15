import os, json, random
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

""" 
Resumen archivos guardados 
* Imágenes transformadas: Data/Data-Transformed/cancer/
* Imágenes descartadas: Data/Data-Discarded/
* Modelo entrenado: model/model_{model}/model_ChestCT.pth
* Logs de entrenamiento: evaluation/evaluation_{model}/training_log_{model}_{fecha}.csv
* Gráfica de pérdidas: evaluation/evaluation_{model}/training_losses_{current_time}_{model}.png
* Imagen generada para evaluar LPIPS: Data/images/images_{model}/img_eval_lpips.png
* Reporte de evaluación: evaluation/evaluation_{model}/EvalModel_{model}_{date}.md
* images/images_{model}/Synthetic_{model}_{i + 1}_{current_date}.png
    
"""

def load_config():
    with open('GAN_PyTorch/config.json', 'r') as json_file:
        return json.load(json_file)

def generate_report_pdf(data_transformed="Data/Data-Transformed/cancer/", 
                         data_discarded="Data/Data-Discarded/", 
                         model="dcgan", 
                         model_trained="model/model_dcgan/model_ChestCT.pth", 
                         log_training="", 
                         graph_training="", 
                         img_to_eval="", 
                         report_eval="", 
                         image_path="", 
                         filename="reporte_archivos.pdf"):
    """
    Genera un PDF con un resumen de los archivos guardados.

    Parámetros:
        - data_transformed: Ruta de imágenes transformadas
        - data_discarded: Ruta de imágenes descartadas
        - model: Nombre del modelo
        - model_trained: Ruta del modelo entrenado
        - log_training: Ruta del log de entrenamiento
        - graph_training: Ruta de la gráfica de pérdidas
        - img_to_eval: Ruta de la imagen evaluada para LPIPS
        - report_eval: Ruta del reporte de evaluación
        - image_path: Ruta de las imágenes generadas
        - filename: Nombre del PDF de salida
    """
    
    # Crear PDF
    c = canvas.Canvas(filename, pagesize=A4)
    
    # Configurar título
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "Resumen archivos guardados")

    # Configurar fuente del contenido
    c.setFont("Helvetica", 12)

    # Lista de líneas a incluir en el PDF
    lines = [
        f"* Imágenes transformadas: {data_transformed}",
        f"* Imágenes descartadas: {data_discarded}",
        f"* Modelo: {model}",
        f"* Modelo entrenado: {model_trained}",
        f"* Logs de entrenamiento: {log_training}" if log_training else "* Logs de entrenamiento: No disponible",
        f"* Gráfica de pérdidas: {graph_training}" if graph_training else "* Gráfica de pérdidas: No disponible",
        f"* Imagen generada para evaluar LPIPS: {img_to_eval}" if img_to_eval else "* Imagen LPIPS: No disponible",
        f"* Reporte de evaluación: {report_eval}" if report_eval else "* Reporte de evaluación: No disponible",
        f"* Carpeta de imágenes generadas: {image_path}" if image_path else "* Imagen generada: No disponible"
    ]

    # Escribir líneas en el PDF
    y_position = 780
    for line in lines:
        c.drawString(100, y_position, line)
        y_position -= 20  # Espaciado entre líneas

    # Añadir imágenes al PDF
    img_y_position = y_position - 40  # Espaciado antes de la primera imagen
    img_width, img_height = 300, 200  # Tamaño de imágenes en el PDF

    def add_image(img_path, y_pos):
        """ Añade una imagen al PDF si el archivo existe """
        if os.path.exists(img_path):
            try:
                img = ImageReader(img_path)
                c.drawImage(img, 100, y_pos, width=img_width, height=img_height)
                return y_pos - img_height - 20  # Ajuste de posición para la siguiente imagen
            except:
                print(f"⚠️ No se pudo cargar la imagen: {img_path}")
        return y_pos  # No cambia la posición si no se carga la imagen

    # Insertar las imágenes si existen
    if graph_training:
        img_y_position = add_image(graph_training, img_y_position)

    if img_to_eval:
        img_y_position = add_image(img_to_eval, img_y_position)

    # Seleccionar una imagen aleatoria de la carpeta de imágenes generadas
    if image_path and os.path.exists(image_path) and os.path.isdir(image_path):
        image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            random_image = os.path.join(image_path, random.choice(image_files))
            img_y_position = add_image(random_image, img_y_position)

    # Guardar PDF
    c.save()
    print(f"📄 PDF generado: {filename}")
    return filename