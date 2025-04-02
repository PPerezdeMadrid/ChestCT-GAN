import os
import json
from fpdf import FPDF
from datetime import datetime

def load_config():
    with open('GAN_PyTorch/config.json', 'r') as json_file:
        return json.load(json_file)

def classify_metric(value, metric_type):
    """ Clasifica la métrica como buena, media o mala """
    if metric_type == 'accuracy':
        if value >= 0.9:
            return "Bueno"
        elif value >= 0.7:
            return "Medio"
        else:
            return "Malo"
    elif metric_type == 'ssim':
        if value >= 0.9:
            return "Bueno"
        elif value >= 0.7:
            return "Medio"
        else:
            return "Malo"
    elif metric_type == 'psnr':
        if value >= 30:
            return "Bueno"
        elif value >= 25:
            return "Medio"
        else:
            return "Malo"
    elif metric_type == 'lpips':
        if value <= 0.2:
            return "Bueno"
        elif value <= 0.5:
            return "Medio"
        else:
            return "Malo"

def validate_eval(accuracy_discriminator, accuracy_generator, ssim_score, psnr_score, lpips_score):
    """Calcula una puntuación del 1 y 10 basado en las métricas del modelo"""

    score_discriminator = (accuracy_discriminator * 100) / 10
    score_generator = (accuracy_generator * 100) / 10

    score_ssim = ssim_score * 10
    
    if psnr_score >= 30:
        score_psnr = 10
    elif psnr_score >= 25:
        score_psnr = 8
    elif psnr_score >= 20:
        score_psnr = 6
    else:
        score_psnr = 4

    score_lpips = max(0, 10 - (lpips_score * 20))  # Invertir LPIPS: cuanto menor, mejor

    # Calificación final: Promedio de todas las puntuaciones
    puntuacion_total = (score_discriminator + score_generator + score_ssim + score_psnr + score_lpips) / 5
    puntuacion_final = round(puntuacion_total)
    
    return puntuacion_final

def generate_report_pdf(data_transformed="Data/Data-Transformed/cancer/", 
                         data_discarded="Data/Data-Discarded/", 
                         model="dcgan", 
                         model_trained="model/model_dcgan/model_ChestCT.pth", 
                         log_training="evaluation/evaluation_dcgan/training_log_dcgan_2025-02-17.csv", 
                         graph_training="evaluation/evaluation_dcgan/training_losses_2025-02-17_19-08-19_dcgan.png", 
                         img_to_eval="evaluation/evaluation_dcgan/img_eval_lpips.png", 
                         report_eval="evaluation/evaluation_dcgan/EvalModel_dcgan_2025-02-17.md", 
                         image_path="images/images_dcgan/Synthetic_dcgan_1_2025-02-17.png", 
                         filename="reporte_archivos.pdf",accuracy_discriminator=0.85, 
                         accuracy_generator=None, 
                         ssim_score=None, 
                         psnr_score=None, 
                         lpips_score=None,):
    """
    Genera un PDF con el reporte de evaluación y la puntuación final.

    Parámetros:
        - model_type: Tipo de modelo (e.g., "dcgan")
        - accuracy_discriminator: Precisión del discriminador
        - accuracy_generator: Precisión del generador
        - ssim_score: SSIM
        - psnr_score: PSNR
        - lpips_score: LPIPS
        - filename: Nombre del PDF de salida
    """
    # Generar la puntuación final
    puntuacion_final = validate_eval(accuracy_discriminator, accuracy_generator, ssim_score, psnr_score, lpips_score)
    
    # Crear PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Título
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, 'Reporte de Evaluación del Modelo GAN', ln=True, align='C')

    # Fecha del reporte
    current_date = datetime.now().strftime('%Y-%m-%d')
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, f"Fecha: {current_date}", ln=True, align='C')
    pdf.ln(10)
    

    pdf.set_font('Arial', '', 12)
    pdf.ln(10)
    if accuracy_discriminator is not None:
        pdf.cell(200, 10, f"Precisión del Discriminador: {accuracy_discriminator * 100:.2f}% ({classify_metric(accuracy_discriminator, 'accuracy')})", ln=True)
    if accuracy_generator is not None:
        pdf.cell(200, 10, f"Precisión del Generador: {accuracy_generator * 100:.2f}% ({classify_metric(accuracy_generator, 'accuracy')})", ln=True)
    if ssim_score is not None:
        pdf.cell(200, 10, f"SSIM: {ssim_score:.4f} ({classify_metric(ssim_score, 'ssim')})", ln=True)
    if psnr_score is not None:
        pdf.cell(200, 10, f"PSNR: {psnr_score:.2f} dB ({classify_metric(psnr_score, 'psnr')})", ln=True)
    if lpips_score is not None:
        pdf.cell(200, 10, f"LPIPS: {lpips_score:.4f} ({classify_metric(lpips_score, 'lpips')})", ln=True)
    pdf.cell(200, 10, f"Puntuación Final: {puntuacion_final}/10", ln=True)
        

    # Gráfica de pérdidas
    if os.path.exists(graph_training):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Gráfica de Pérdidas del Entrenamiento", ln=True, align='C')
        pdf.image(graph_training, x=10, y=pdf.get_y(), w=180)
        pdf.ln(90)  

   # Imagen generada
    if os.path.exists(img_to_eval):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Imagen Generada para Evaluación", ln=True, align='C')
        
        # Calcular la posición X para centrar la imagen
        img_width = 30  # Ancho de la imagen
        x_pos = (pdf.w - img_width) / 2  # Centro en la página
        
        pdf.image(img_to_eval, x=x_pos, y=pdf.get_y(), w=img_width)
        pdf.ln(45)  
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 11, "Esta es la imagen generada por el modelo para evaluar si merece la pena o no generar más imágenes.", align='C')
        pdf.ln(10)


    # Título H1
    pdf.set_font('Arial', 'B', 16) 
    pdf.cell(200, 10, f'Modelo: {model}', ln=True, align='C')

    # Información general sobre el modelo y los archivos
    pdf.set_font("Arial", 'B', 12)  # Negrita para las etiquetas
    pdf.multi_cell(0, 10, f"Ruta del modelo entrenado: ", align='L')
    pdf.set_font("Arial", size=12)  # Texto normal para el valor
    pdf.multi_cell(0, 10, f"{model_trained}", align='L')

    pdf.set_font("Arial", 'B', 12)
    pdf.multi_cell(0, 10, f"Log de entrenamiento: ", align='L')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{log_training}", align='L')

    pdf.set_font("Arial", 'B', 12)
    pdf.multi_cell(0, 10, f"Gráfica de pérdidas: ", align='L')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{graph_training}", align='L')

    pdf.set_font("Arial", 'B', 12)
    pdf.multi_cell(0, 10, f"Imagen para evaluar LPIPS: ", align='L')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{img_to_eval}", align='L')

    pdf.set_font("Arial", 'B', 12)
    pdf.multi_cell(0, 10, f"Reporte de evaluación: ", align='L')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{report_eval}", align='L')

    pdf.set_font("Arial", 'B', 12)
    pdf.multi_cell(0, 10, f"Imagen generada: ", align='L')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{image_path}", align='L')

    pdf.set_font("Arial", 'B', 12)
    pdf.multi_cell(0, 10, f"Imágenes transformadas: ", align='L')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{data_transformed}", align='L')

    pdf.set_font("Arial", 'B', 12)
    pdf.multi_cell(0, 10, f"Imágenes descartadas: ", align='L')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{data_discarded}", align='L')

    pdf.ln(10)

    # Guardar el PDF
    pdf.output(filename)
    print(f"PDF generado: {filename}")

""" Ejemplo de uso
model_type = "dcgan"
accuracy_discriminator = 0.85
accuracy_generator = 0.88
ssim_score = 0.92
psnr_score = 28.5
lpips_score = 0.3

generate_report_pdf(accuracy_discriminator=accuracy_discriminator,
                    accuracy_generator=accuracy_generator,
                    ssim_score=ssim_score,
                    psnr_score=psnr_score,
                    lpips_score=lpips_score)    
"""