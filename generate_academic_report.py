"""
Script para generar el Informe Académico Técnico Completo en formato Microsoft Word (.docx).
Cumple con todos los capítulos, ecuaciones, tablas exactas y figuras del proyecto.
"""
import os
import sys
import json
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

# Paths
BASE_DIR = r"c:\Users\Asus\Desktop\Programas\IA2\Proyecto_LEGO"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DOCX_OUT = os.path.join(BASE_DIR, "Informe_Academico_LEGO_Vision.docx")

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set cell padding in twips (1/20 of a pt)."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    """Set cell background color."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), color_hex)
    tcPr.append(shd)

def create_document():
    doc = Document()
    
    # Page setup (Standard Letter, 1 inch margins)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.different_first_page_header_footer = True
        
        # Header & Footer
        footer = section.footer
        f_p = footer.paragraphs[0]
        f_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        f_run = f_p.add_run("Proyecto LEGO: Segmentación y Clasificación | Informe Técnico")
        f_run.font.name = "Calibri"
        f_run.font.size = Pt(8.5)
        f_run.font.color.rgb = RGBColor(128, 128, 128)

    # Styles setup
    style_normal = doc.styles['Normal']
    font_normal = style_normal.font
    font_normal.name = 'Calibri'
    font_normal.size = Pt(11)
    font_normal.color.rgb = RGBColor(35, 35, 35)

    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(36)
        p.paragraph_format.space_after = Pt(12)
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(22)
        run.font.color.rgb = RGBColor(26, 54, 93) # Deep Blue

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(24)
        run = p.add_run(text)
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(74, 85, 104)

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(20)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(15)
        run.font.color.rgb = RGBColor(31, 78, 121)

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(12.5)
        run.font.color.rgb = RGBColor(43, 108, 176)

    def add_h3(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(11.5)
        run.font.color.rgb = RGBColor(74, 85, 104)

    def add_p(text, bold_prefix=None, space_after=6, italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.bold = True
            r_pre.font.color.rgb = RGBColor(31, 78, 121)
        r = p.add_run(text)
        r.italic = italic
        return p

    def add_callout(text, title="NOTA TÉCNICA"):
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        set_cell_shading(cell, "F0F4F8") # soft light blue
        set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.15
        r_title = p.add_run(f"[{title}] ")
        r_title.bold = True
        r_title.font.color.rgb = RGBColor(31, 78, 121)
        r_text = p.add_run(text)
        r_text.font.size = Pt(10)
        doc.add_paragraph().paragraph_format.space_after = Pt(6)

    def add_image_if_exists(img_rel_path, caption=None, width=Inches(5.8)):
        full_path = os.path.join(BASE_DIR, img_rel_path)
        if os.path.exists(full_path):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            doc.add_picture(full_path, width=width)
            if caption:
                cp = doc.add_paragraph()
                cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cp.paragraph_format.space_after = Pt(12)
                r = cp.add_run(caption)
                r.italic = True
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(100, 100, 100)

    # =========================================================================
    # PORTADA ACADÉMICA
    # =========================================================================
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_inst = p_inst.add_run("UNIVERSIDAD / INSTITUCIÓN ACADÉMICA\nFACULTAD DE INGENIERÍA Y CIENCIAS APLICADAS\nDEPARTAMENTO DE INTELIGENCIA ARTIFICIAL Y COMPUTACIÓN")
    r_inst.bold = True
    r_inst.font.size = Pt(11)
    r_inst.font.color.rgb = RGBColor(70, 70, 70)
    
    add_title("SEGMENTACIÓN Y CLASIFICACIÓN DE PIEZAS LEGO MEDIANTE MACHINE LEARNING Y DEEP LEARNING")
    add_subtitle("Diseño, Experimentación y Comparación de Algoritmos Tradicionales de Aprendizaje Automático frente a Redes Neuronales Convolucionales sobre un Catálogo de 20 Clases")

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_before = Pt(80)
    p_meta.paragraph_format.space_after = Pt(40)
    r_meta = p_meta.add_run("Autor / Investigador: Julio Alfonso Curiel García\n"
                           "Área: Visión por Computador y Reconocimiento de Patrones\n"
                           "Entorno Experimental: Python 3.13, PyTorch, OpenCV, Scikit-Learn\n"
                           "Fecha: Noviembre 2026")
    r_meta.font.size = Pt(11)
    r_meta.font.color.rgb = RGBColor(80, 80, 80)

    doc.add_page_break()

    # =========================================================================
    # RESUMEN Y ABSTRACT
    # =========================================================================
    add_h1("RESUMEN EJECUTIVO")
    add_p("El presente informe documenta de manera exhaustiva el desarrollo, evaluación empírica y despliegue de un sistema de visión artificial integral diseñado para la localización espacial precisa, segmentación y clasificación automática de piezas LEGO. La investigación aborda un dataset cerrado de 20 categorías divididas en las familias Brick y Plate, comprendiendo un total de 4,580 imágenes recortadas y 3,464 fotografías de escena base capturadas en condiciones de iluminación retroiluminada. "
          "Para la etapa de segmentación, se implementó un pipeline en OpenCV basado en la separación de canales en el espacio de color HSV (umbralización adaptativa de Otsu sobre el canal de saturación S y umbral inverso sobre el canal de luminancia V), seguido de operadores morfológicos y análisis de contornos con margen de guarda. La calidad de segmentación fue validada cuantitativamente sobre un benchmark con máscaras ground-truth, alcanzando un Recall de 95.16%, Pixel Accuracy de 81.92%, Intersection over Union (IoU) de 0.6457 y Coeficiente Dice de 0.7011.")
    add_p("Para la clasificación multiclase, se entrenaron y evaluaron bajo condiciones rigurosas y libres de fuga de datos (Data Leakage) seis algoritmos representativos: Perceptrón, SGAdaline (Adaline con Stochastic Gradient Descent y pérdida MSE), Support Vector Machines (SVM), Árboles de Decisión (CART), Redes Neuronales Artificiales densas (ANN/MLP) y Redes Neuronales Convolucionales (CNN con Data Augmentation). "
          "Asimismo, se contrastaron empíricamente dos espacios de representación: 'Solo Forma y Geometría' (301 dimensiones, integrando Hu Moments, medidas de solidez, compacidad y descriptores HOG) versus 'Forma, Geometría y Color' (325 dimensiones con histogramas HSV). Los resultados experimentales en el conjunto de prueba independiente confirmaron que la Red Neuronal Convolucional (CNN) es el modelo ganador absoluto con 95.63% de Exactitud y 95.55% de F1-Score Macro. Por su parte, la Support Vector Machine con kernel RBF se consagró como la alternativa de óptimo equilibrio computacional, logrando 90.83% de Exactitud con apenas 12.01 segundos de entrenamiento y 1.27 ms de latencia por muestra.")
    
    add_p("Palabras clave: Visión por computador, Segmentación de imágenes, OpenCV, Espacio HSV, LEGO Bricks, Machine Learning, Deep Learning, HOG, Hu Moments, CNN, SVM.", bold_prefix="Descriptores: ")

    doc.add_page_break()

    # =========================================================================
    # CAPÍTULO 1: INTRODUCCIÓN
    # =========================================================================
    add_h1("CAPÍTULO 1. INTRODUCCIÓN")
    
    add_h2("1.1. Contextualización")
    add_p("La visión por computador es una disciplina consolidada en la intersección de la informática, el procesamiento de señales y el aprendizaje automático, cuyo propósito primordial es emular la capacidad del sistema visual humano para interpretar, estructurar y extraer significado de fuentes visuales bidimensionales. En los entornos industriales, logísticos y de automatización robótica contemporánea, la inspección automática y el ensamblaje inteligente de piezas mecánicas requieren sistemas de alta fidelidad capaces de discernir sutiles variaciones morfológicas entre componentes de dimensiones estandarizadas.")
    add_p("En este marco, las piezas del sistema de construcción LEGO representan un entorno de prueba idóneo y de alta complejidad para la visión artificial. A pesar de su apariencia lúdica, las piezas LEGO poseen geometrías milimétricamente calibradas basadas en una cuadrícula modular (determinada por la disposición periódica de tetones o 'studs' cilíndricos superiores) y proporciones de altura fijas entre las familias de bloques gruesos ('Bricks') y delgados ('Plates'). La automatización del reconocimiento de estas piezas abre la puerta a soluciones industriales escalables en reciclaje automatizado, clasificación de piezas en líneas de manufactura y robótica de pick-and-place.")

    add_h2("1.2. Planteamiento del Problema")
    add_p("El problema técnico a resolver radica en el diseño de un sistema capaz de recibir una fotografía de escena, segmentar de forma no destructiva las regiones donde se encuentran las piezas y clasificar cada instancia detectada dentro de una de las 20 categorías del catálogo disponible. El logro de este objetivo enfrenta retos visuales específicos:")
    add_p("• Variaciones Cromáticas Severas: Una misma categoría de pieza (por ejemplo, Brick 2x4) puede estar fabricada en plástico azul, rojo, amarillo, verde, marrón o negro. El sistema debe evitar asociar la categoría a un color específico (sesgo cromático), reconociendo la identidad por su estructura topológica.", space_after=4)
    add_p("• Similitudes Geométricas entre Categorías Vecinas: Ciertas clases presentan discrepancias dimensionales mínimas. Por ejemplo, distinguir entre un Plate 1x2 y un Plate 1x3, o entre un Plate 2x4 y un Brick 2x4 desde ciertos ángulos, exige capturar la relación de aspecto y la altura con extrema precisión.", space_after=4)
    add_p("• Perspectiva, Rotación y Oclusión: Las piezas pueden colocarse en ángulos de rotación arbitrarios en el plano ($0^\circ$ a $360^\circ$) o en perspectivas oblicuas, modificando la proyección aparente de los tetones y las caras laterales.", space_after=4)
    add_p("• Condiciones de Fondo e Iluminación: Sombras proyectadas, reflejos especulares sobre el plástico brillante y artefactos producidos por mesas de iluminación plantean falsas discontinuidades de borde.", space_after=4)
    add_p("• Desafío de Instancias Múltiples y Contacto: Cuando dos piezas se tocan en la imagen, los contornos clásicos tienden a unificarlas en un único objeto espurio, requiriendo algoritmos de desacoplamiento como la transformada de distancia y Watershed.", space_after=6)

    add_h2("1.3. Justificación")
    add_p("La integración coordinada entre técnicas de Procesamiento Digital de Imágenes (PDI) mediante OpenCV y modelos de clasificación supervisada permite construir una arquitectura desacoplada y robusta. Mientras que los métodos de PDI aíslan la región de interés (ROI) reduciendo drásticamente la dimensionalidad del problema y eliminando información ruidosa del entorno, los modelos de Machine Learning y Deep Learning se concentran en discriminar los patrones estructurales locales. Evaluar simultáneamente seis modelos paradigmáticos (lineales simples, optimización estocástica, árboles no paramétricos, funciones de base radial y convoluciones profundas) permite identificar de manera científicamente fundada la frontera de compromiso óptima entre exactitud estadística y costo de cómputo en inferencia.")

    add_h2("1.4. Objetivo General")
    add_p("Desarrollar, entrenar, evaluar y comparar experimentalmente un sistema modular de visión por computador para la segmentación, localización y clasificación multiclase de piezas LEGO sobre un dataset estructurado de 20 categorías, contrastando modelos clásicos de Machine Learning frente a arquitecturas neuronales profundas mediante métricas cuantitativas reproducibles.")

    add_h2("1.5. Objetivos Específicos")
    add_p("1. Explorar, depurar y cuantificar la distribución del dataset LEGO, verificando integridad física, formatos, resoluciones y correlaciones unívocas entre imágenes base y recortes para prevenir estrictamente la fuga de datos (Data Leakage).", space_after=3)
    add_p("2. Diseñar un pipeline de segmentación en OpenCV que opere en el espacio HSV combinando umbrales duales de Otsu (canales S y V), operadores morfológicos y filtrado perimetral con margen de seguridad.", space_after=3)
    add_p("3. Validar cuantitativamente las máscaras binarias generadas frente a referencias anotadas mediante IoU, Coeficiente Dice, Pixel Accuracy, Precision y Recall.", space_after=3)
    add_p("4. Extraer vectores descriptores numéricos basados en características geométricas normalizadas, 7 momentos invariantes de Hu, histogramas de gradientes orientados (HOG) e histogramas cromáticos HSV, comparando los escenarios de Solo Forma vs. Forma+Color.", space_after=3)
    add_p("5. Implementar y sintonizar rigurosamente los seis modelos: Perceptrón, SGAdaline multiclase con descenso de gradiente estocástico, Support Vector Machine (evaluando kernels lineal, polinómico y RBF), Árboles de Decisión (CART), Red Neuronal Artificial multicapa (ANN/MLP) y Red Neuronal Convolucional (CNN con Data Augmentation).", space_after=3)
    add_p("6. Evaluar cuantitativamente los clasificadores en un conjunto de prueba estratificado idéntico, calculando matrices de confusión, curvas de pérdida/exactitud, tiempos de entrenamiento y latencia de inferencia.", space_after=3)
    add_p("7. Integrar las etapas de segmentación OpenCV y el modelo clasificador óptimo en un pipeline interactivo unificado capaz de recibir fotografías y producir paneles de diagnóstico visual.", space_after=6)

    add_h2("1.6. Alcance y Limitaciones")
    add_p("Alcance implementado: El sistema opera de extremo a extremo cubriendo las 20 categorías del dataset con modelos entrenados y persistidos en disco, un pipeline funcional de inferencia, una aplicación interactiva desarrollada en Streamlit, un cuaderno reproducible de Google Colab / Jupyter y una suite de pruebas automatizadas con pytest.")
    add_p("Limitaciones identificadas: El pipeline de segmentación OpenCV asume fondos relativamente homogéneos o claros (como la mesa retroiluminada provista en el dataset o papel blanco liso). En escenarios del mundo real con fondos texturalmente complejos, manos humanas sosteniendo la pieza o fondos oscuros heterogéneos, la segmentación clásica por umbral cromático global requiere filtros complementarios (como el filtro de piel en YCrCb incorporado en etapas avanzadas) o el futuro reemplazo por redes neuronales de segmentación de instancias (Mask R-CNN / YOLO-seg).")

    # =========================================================================
    # CAPÍTULO 2: MARCO TEÓRICO
    # =========================================================================
    add_h1("CAPÍTULO 2. MARCO TEÓRICO")

    add_h2("2.1. Visión por Computador y Procesamiento Digital de Imágenes")
    add_p("Una imagen digital en color de dimensiones $W \times H$ píxeles se modela formalmente como una función discreta $I: \mathbb{Z}^2 \to \mathbb{R}^C$, donde $C$ representa el número de canales espectrales. En el espacio estándar de adquisición BGR (Blue, Green, Red), cada píxel almacena una tupla de intensidades cuantizadas típicamente en 8 bits ($[0, 255]$):")
    add_p("$$I(x, y) = [I_B(x, y), I_G(x, y), I_R(x, y)]^T \in [0, 255]^3$$", italic=True)
    add_p("El procesamiento digital de imágenes abarca las transformaciones algebraicas y estadísticas sobre esta matriz discreta con el fin de mejorar su relación señal-ruido, extraer bordes geométricos y transformar la señal continua del mundo real en información estructurada interpretable por máquinas.")

    add_h2("2.2. Espacio de Color HSV")
    add_p("El espacio de color BGR presenta una correlación no lineal intrínseca entre la luminancia y la cromaticidad, lo que hace que variaciones en la intensidad lumínica alteren simultáneamente los tres canales. Para mitigar esta debilidad en tareas de segmentación por color, se realiza la conversión al espacio HSV (Hue, Saturation, Value):")
    add_p("• Matiz (Hue, $H$): Codifica la longitud de onda dominante o tono del color como un ángulo en el círculo cromático ($[0^\circ, 360^\circ]$, mapeado en OpenCV a $[0, 180]$ para almacenar en un byte uint8). Permite discriminar rojo, amarillo, verde, azul o cian con independencia de la iluminación.", space_after=3)
    add_p("• Saturación (Saturation, $S$): Cuantifica la pureza o viveza cromática ($[0, 255]$). Los colores plásticos vivos poseen alta saturación ($S > 120$), mientras que los blancos, grises y fondos translúcidos presentan baja saturación ($S < 80$).", space_after=3)
    add_p("• Valor (Value, $V$): Expresa la luminancia o brillo intrínseco de la escena ($[0, 255]$). Los píxeles negros y sombras profundas se concentran en valores reducidos ($V < 100$).", space_after=6)

    add_h2("2.3. Técnicas de Umbralización y Método de Otsu")
    add_p("La umbralización binariza una imagen en escala de grises $g(x, y)$ mapeándola a una máscara binaria $M(x, y) \in \{0, 1\}$ mediante un valor umbral $T$:")
    add_p("$$M(x, y) = \\begin{cases} 1 & \\text{si } g(x, y) \\ge T \\\\ 0 & \\text{en caso contrario} \\end{cases}$$", italic=True)
    add_p("El algoritmo de Otsu determina de manera óptima y no paramétrica el umbral $T^*$ maximizando la varianza entre clases $\\sigma_B^2(T)$, lo cual equivale a minimizar la varianza intra-clase ponderada $\\sigma_W^2(T)$ sobre el histograma normalizado de probabilidades $p_i$ ($i \\in [0, L-1]$):")
    add_p("$$\\sigma_B^2(T) = \\omega_0(T) [\\mu_0(T) - \\mu_T]^2 + \\omega_1(T) [\\mu_1(T) - \\mu_T]^2$$", italic=True)
    add_p("donde $\\omega_0(T)$ y $\\omega_1(T)$ son las probabilidades acumuladas del fondo y del primer plano, $\\mu_0(T)$ y $\\mu_1(T)$ sus medias respectivas, y $\\mu_T$ la media global del histograma.")

    add_h2("2.4. Operaciones Morfológicas Matemáticas")
    add_p("La morfología matemática opera sobre conjuntos en el espacio discreto utilizando un elemento estructurante $B$ (kernel) de tamaño predefinido:")
    add_p("• Erosión ($A \\ominus B$): Elimina píxeles aislados y reduce las dimensiones perimetrales del objeto:")
    add_p("$$A \\ominus B = \\{ z \\in \\mathbb{Z}^2 \\mid (B)_z \\subseteq A \\}$$", italic=True)
    add_p("• Dilatación ($A \\oplus B$): Expande las fronteras del objeto y reconecta componentes discontinuas:")
    add_p("$$A \\oplus B = \\{ z \\in \\mathbb{Z}^2 \\mid (\\hat{B})_z \\cap A \\ne \\emptyset \\}$$", italic=True)
    add_p("• Apertura Morfológica ($A \\circ B = (A \\ominus B) \\oplus B$): Suprime protuberancias delgadas, espículas y ruido puntual sin alterar significativamente el área global.", space_after=3)
    add_p("• Cierre Morfológico ($A \\bullet B = (A \\oplus B) \\ominus B$): Sella pequeños agujeros internos, fisuras y sombras producidas por los tetones superiores de las piezas.", space_after=6)

    add_h2("2.5. Análisis de Contornos y Geometría")
    add_p("La detección de bordes y contornos se basa en el algoritmo topológico de Suzuki y Abe (implementado en cv2.findContours), el cual rastrea las transiciones $0 \\to 1$ extrayendo las curvas poligonales perimetrales $C = \\{p_1, p_2, \\dots, p_N\\}$. A partir del contorno se calculan los momentos espaciales $m_{pq}$ y momentos centrales $\\mu_{pq}$:")
    add_p("$$m_{pq} = \\sum_{x} \\sum_{y} x^p y^q I(x, y), \\quad \\mu_{pq} = \\sum_{x} \\sum_{y} (x - \\bar{x})^p (y - \\bar{y})^q I(x, y)$$", italic=True)
    add_p("El área de Green se obtiene como $A = m_{00}$ y el perímetro $P$ integrando la longitud de los segmentos perimetrales.")

    add_h2("2.6. Extracción de Características")
    add_p("Para los clasificadores estadísticos tradicionales se emplean tres grupos fundamentales de descriptores:")
    add_p("1. Descriptores Geométricos Normalizados: Relación de aspecto ($AR = w/h$), extensión ($A / (w \\cdot h)$), solidez ($A / A_{\\text{convex}}$), compacidad circular ($4\\pi A / P^2$), área normalizada por la resolución y perímetro normalizado.", space_after=3)
    add_p("2. Momentos Invariantes de Hu: Conjunto de 7 momentos ortogonales calculados a partir de los momentos centrales normalizados $\\eta_{pq} = \\mu_{pq} / \\mu_{00}^{1 + (p+q)/2}$. Son matemáticamente invariantes frente a traslaciones, escalado y rotaciones bidimensionales completas. Se transforman logarítmicamente preservando el signo: $h_i^{\\text{log}} = -\\text{sign}(h_i) \\log_{10}(|h_i| + 10^{-10})$.", space_after=3)
    add_p("3. Histograma de Gradientes Orientados (HOG - Dalal & Triggs): Mide la distribución local de intensidades de gradiente y orientaciones en celdas de $16 \\times 16$ píxeles, agrupadas en bloques normalizados L2-Hys de $2 \\times 2$ celdas con 8 orientaciones angulares. Captura fielmente la densidad y disposición espacial de los tetones de los LEGO.", space_after=3)
    add_p("4. Histogramas de Color HSV: Descriptores de densidad de probabilidad normalizados L1 para los canales $H$, $S$ y $V$ cuantizados en 8 bins por canal (24 características en total).", space_after=6)

    add_h2("2.7. Fundamentos de los Algoritmos de Clasificación")
    add_p("A continuación se sintetizan los seis modelos evaluados en la investigación:")
    
    add_h3("A. Perceptrón de Rosenblatt")
    add_p("El clasificador lineal primigenio modela la frontera como $f(x) = \\text{sign}(w^T x + b)$. Su aprendizaje actualiza los pesos de forma aditiva únicamente cuando se produce un error de clasificación: $w \\leftarrow w + \\eta (y_i - \\hat{y}_i) x_i$. Su convergencia está garantizada únicamente si el problema es linealmente separable (Teorema del Perceptrón de Novikoff). Para clasificar 20 clases, se descompone en un esquema One-vs-Rest (OvR).")

    add_h3("B. SGAdaline (Adaptive Linear Neuron con SGD)")
    add_p("A diferencia del Perceptrón clásico que actualiza según la salida discontinua de paso, Adaline (Widrow & Hoff) aprende sobre la salida lineal continua $\\hat{z}_i = w^T x_i + b$ minimizando la función de costo del Error Cuadrático Medio (MSE):")
    add_p("$$J(w, b) = \\frac{1}{2N} \\sum_{i=1}^N (y_i - (w^T x_i + b))^2$$", italic=True)
    add_p("Bajo Stochastic Gradient Descent (SGD), los pesos se corrigen muestra a muestra barajando los datos en cada época con una tasa con decaimiento:")
    add_p("$$w \\leftarrow w + \\eta (y_i - \\hat{z}_i) x_i, \\quad b \\leftarrow b + \\eta (y_i - \\hat{z}_i), \\quad \\eta_t = \\frac{\\eta_0}{1 + \\lambda t}$$", italic=True)

    add_h3("C. Support Vector Machines (SVM)")
    add_p("SVM (Cortes & Vapnik) construye el hiperplano óptimo que maximiza el margen geométrico $2 / \\|w\\|$ entre las clases más cercanas (vectores de soporte), controlando la penalización por margen suave mediante el parámetro de regularización $C$:")
    add_p("$$\\min_{w, b, \\xi} \\frac{1}{2} \\|w\\|^2 + C \\sum_{i=1}^N \\xi_i \\quad \\text{s.a. } y_i(w^T \\phi(x_i) + b) \\ge 1 - \\xi_i, \\quad \\xi_i \\ge 0$$", italic=True)
    add_p("El kernel trick permite proyectar implícitamente las características a un espacio de Hilbert de dimensión superior sin computar explícitamente las coordenadas. Se evalúan el kernel lineal $K(x, x') = x^T x'$, el polinómico $K(x, x') = (\\gamma x^T x' + 1)^d$ y la función de base radial Gaussiana (RBF):")
    add_p("$$K_{\\text{RBF}}(x, x') = \\exp(-\\gamma \\|x - x'\\|^2)$$", italic=True)

    add_h3("D. Árboles de Decisión (CART)")
    add_p("El algoritmo CART (Breiman et al.) divide recursivamente el espacio de características mediante hiperplanos ortogonales a los ejes coordenados. En cada nodo interno, selecciona el umbral $t$ sobre la característica $j$ que maximiza la reducción de impureza de Gini:")
    add_p("$$I_G(S) = 1 - \\sum_{k=1}^K p_k^2, \\quad \\Delta I_G = I_G(S) - \\frac{|S_L|}{|S|} I_G(S_L) - \\frac{|S_R|}{|S|} I_G(S_R)$$", italic=True)

    add_h3("E. Redes Neuronales Artificiales (ANN / MLP)")
    add_p("El perceptrón multicapa densamente conectado aproxima cualquier función continua (Teorema de Aproximación Universal de Cybenko). Cada capa oculta computa $a^{(l)} = \\sigma(W^{(l)} a^{(l-1)} + b^{(l)})$, aplicando funciones de activación no lineales como ReLU ($f(z) = \\max(0, z)$). Los pesos se actualizan mediante el optimizador Adam retropropagando el gradiente de la entropía cruzada multiclase:")
    add_p("$$\\mathcal{L}_{\\text{CE}} = -\\sum_{k=1}^K y_k \\log(\\hat{p}_k), \\quad \\hat{p}_k = \\frac{\\exp(z_k)}{\\sum_j \\exp(z_j)}$$", italic=True)

    add_h3("F. Redes Neuronales Convolucionales (CNN)")
    add_p("Las CNNs explotan la correlación espacial y la invarianza traslacional de las imágenes aplicando bancos de filtros convolucionales parametrizados $K$ de dimensiones $k_w \\times k_h$ que operan mediante producto tensorial local:")
    add_p("$$S(i, j) = (I * K)(i, j) = \\sum_{m} \\sum_{n} I(i-m, j-n) K(m, n)$$", italic=True)
    add_p("A diferencia de los modelos clásicos, la CNN elimina la necesidad de diseñar 'a mano' descriptores como HOG o Hu, aprendiendo de manera jerárquica características de bajo nivel (bordes y esquinas en las primeras capas) y de alto nivel (tetones, formas en 'L', rampas y relaciones tridimensionales en capas profundas).")

    add_h2("2.8. Métricas Matemáticas de Evaluación")
    add_p("Para la clasificación multiclase sobre $K=20$ clases, se utilizan las siguientes formulaciones:")
    add_p("$$\\text{Accuracy} = \\frac{\\sum_{k=1}^K TP_k}{\\sum_{k=1}^K (TP_k + FP_k + FN_k + TN_k)}$$", italic=True)
    add_p("$$\\text{Precision}_{\\text{Macro}} = \\frac{1}{K} \\sum_{k=1}^K \\frac{TP_k}{TP_k + FP_k}, \\quad \\text{Recall}_{\\text{Macro}} = \\frac{1}{K} \\sum_{k=1}^K \\frac{TP_k}{TP_k + FN_k}$$", italic=True)
    add_p("$$\\text{F1-Score}_{\\text{Macro}} = \\frac{1}{K} \\sum_{k=1}^K 2 \\cdot \\frac{\\text{Precision}_k \\cdot \\text{Recall}_k}{\\text{Precision}_k + \\text{Recall}_k}$$", italic=True)
    add_p("Para evaluar la calidad de segmentación entre la máscara binaria predicha $M_P$ y la referencia real $M_T$:")
    add_p("$$\\text{IoU (Jaccard)} = \\frac{|M_T \\cap M_P|}{|M_T \\cup M_P|} = \\frac{TP}{TP + FP + FN}$$", italic=True)
    add_p("$$\\text{Dice (F1 Píxel)} = \\frac{2 |M_T \\cap M_P|}{|M_T| + |M_P|} = \\frac{2 TP}{2 TP + FP + FN}$$", italic=True)
    add_p("$$\\text{Pixel Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}$$", italic=True)

    # =========================================================================
    # CAPÍTULO 3: MATERIALES Y METODOLOGÍA
    # =========================================================================
    add_h1("CAPÍTULO 3. MATERIALES Y METODOLOGÍA")

    add_h2("3.1. Entorno de Desarrollo y Bibliotecas")
    add_p("El proyecto se diseñó para operar de forma totalmente reproducible tanto en Google Colab como en entornos locales multiplataforma (Windows 11 / Linux). Las versiones verificadas de las bibliotecas de software empleadas son las siguientes:")
    add_p("• Lenguaje de programación: Python 3.13.7 (64-bit).\n"
          "• OpenCV (opencv-python): Versión 5.0.0.93 para procesamiento digital de imágenes, transformaciones cromáticas, morfología y análisis de contornos.\n"
          "• PyTorch (torch / torchvision): Versión 2.15.0+cpu y torchvision 0.29.2+cpu para la implementación, entrenamiento y Data Augmentation de los modelos neuronales (ANN y CNN).\n"
          "• Scikit-learn (sklearn): Versión 1.9.1 para el particionado estratificado, Perceptrón, SVM, Árboles de Decisión, escalamiento StandardScaler y matrices de confusión.\n"
          "• Scikit-image (skimage): Versión 0.26.0 para la extracción robusta de descriptores HOG (Histogram of Oriented Gradients).\n"
          "• NumPy: Versión 2.3.3 para operaciones de álgebra lineal y matrices multidimensionales.\n"
          "• Pandas: Versión 2.3.3 para estructuración y persistencia de métricas y particiones tabulares.\n"
          "• Matplotlib: Versión 3.10.7 para visualización y generación de gráficas estadísticas.\n"
          "• Streamlit: Versión 1.66.1 para la interfaz de despliegue visual interactiva.", space_after=8)

    add_h2("3.2. Descripción y Estructura Real del Dataset")
    add_p("El dataset original provisto se encuentra dividido en dos carpetas principales:")
    add_p("1. LEGO_BRICKS/Base Images/: Contiene 3,464 fotografías de escena en resolución constante de $640 \\times 480$ píxeles, donde las piezas se encuentran sobre un fondo retroiluminado difuso de tonalidad cian suave. La distribución comprende 15 de las 20 clases.", space_after=3)
    add_p("2. LEGO_BRICKS/Cropped Images/: Contiene 4,580 recortes individuales de piezas centradas, cubriendo de forma balanceada la totalidad de las 20 categorías del problema.", space_after=6)

    # Tabla de distribución real
    add_p("Tabla 1. Distribución real de imágenes por categoría en el dataset LEGO.", bold_prefix="Tabla 1: ", space_after=4)
    table_data = [
        ["Familia", "Categoría LEGO", "Base Images (640x480)", "Cropped Images", "Resolución Promedio (Cropped)"],
        ["Brick", "Brick_1x1", "0*", "216", "66 x 62 px"],
        ["Brick", "Brick_1x2", "0*", "239", "78 x 64 px"],
        ["Brick", "Brick_1x3", "0*", "188", "112 x 64 px"],
        ["Brick", "Brick_1x4", "0*", "238", "148 x 65 px"],
        ["Brick", "Brick_2x2", "0*", "235", "92 x 90 px"],
        ["Brick", "Brick_2x2_L", "317", "317", "136 x 161 px"],
        ["Brick", "Brick_2x2_Slope", "192", "192", "142 x 129 px"],
        ["Brick", "Brick_2x3", "202", "202", "168 x 190 px"],
        ["Brick", "Brick_2x4", "242", "242", "232 x 125 px"],
        ["Plate", "Plate_1x1", "291", "291", "64 x 51 px"],
        ["Plate", "Plate_1x1_Round", "212", "212", "65 x 71 px"],
        ["Plate", "Plate_1x1_Slope", "329", "329", "88 x 76 px"],
        ["Plate", "Plate_1x2", "207", "207", "76 x 52 px"],
        ["Plate", "Plate_1x2_Grill", "217", "217", "78 x 54 px"],
        ["Plate", "Plate_1x3", "199", "199", "110 x 53 px"],
        ["Plate", "Plate_1x4", "223", "223", "145 x 52 px"],
        ["Plate", "Plate_2x2", "207", "207", "90 x 82 px"],
        ["Plate", "Plate_2x2_L", "224", "224", "132 x 155 px"],
        ["Plate", "Plate_2x3", "189", "189", "165 x 150 px"],
        ["Plate", "Plate_2x4", "213", "213", "220 x 110 px"],
        ["TOTAL", "20 Categorías", "3,464", "4,580", "Dimensiones variables"]
    ]
    t = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table_data):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            if i == 0 or i == len(table_data)-1:
                set_cell_shading(cell, "D9E1F2")
                p.runs[0].bold = True
            else:
                set_cell_shading(cell, "FFFFFF" if i%2==0 else "F9FBFD")
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
    
    add_p("*Nota técnica sobre Base Images: Las 5 categorías Brick 1x1, 1x2, 1x3, 1x4 y 2x2 no contaban con fotografías originales en Base Images, pero sí estaban presentes con 1,116 imágenes de alta calidad en Cropped Images. Esto no afectó el entrenamiento de los seis modelos clasificadores sobre las 20 categorías completas.", italic=True, space_after=8)

    add_image_if_exists("results/exploration/class_distribution.png", "Figura 1. Distribución empírica de clases en Cropped Images (Total: 4,580 muestras en 20 categorías).")
    add_image_if_exists("results/exploration/resolution_and_aspect_ratio.png", "Figura 2. Análisis de dispersión de resoluciones (ancho vs alto) y distribución de relaciones de aspecto.")

    add_h2("3.3. Limpieza y Auditoría de Integridad")
    add_p("Mediante el script dataset_explorer.py se aplicó una auditoría exhaustiva:")
    add_p("• Integridad de archivos: 100% de los 8,044 archivos (.jpg) se decodificaron con éxito mediante cv2.imread(), arrojando 0 archivos corruptos o ilegibles.", space_after=3)
    add_p("• Detección de duplicados exactos: El cálculo del hash criptográfico MD5 sobre cada flujo binario confirmó que existen 0 duplicados exactos dentro de las colecciones.", space_after=3)
    add_p("• Correspondencia Base <-> Cropped: Se verificó mediante coincidencia de expresiones regulares que las imágenes base y los recortes comparten la misma marca temporal identificadora (timestamp_id de 12 dígitos, ej. '180708213430'). De las 3,464 imágenes base, 3,462 (99.94%) tienen correspondencia 1 a 1 exacta con un recorte.", space_after=6)

    add_h2("3.4. Control de Fuga de Datos (Data Leakage) y Particionado")
    add_p("Un error metodológico común en problemas de visión es permitir que recortes y escenas base pertenecientes a la misma sesión fotográfica se distribuyan aleatoriamente entre entrenamiento y prueba. Para evitar esta contaminación (Data Leakage), el módulo data_loader.py implementó una partición estratificada agrupada (Group-aware Stratified Split) asegurando que ninguna muestra con el mismo timestamp_id coexista en particiones distintas:")
    add_p("• Conjunto de Entrenamiento (Train): 3,206 muestras (70.0% de los datos).\n"
          "• Conjunto de Validación (Val): 687 muestras (15.0% de los datos).\n"
          "• Conjunto de Prueba Independiente (Test): 687 muestras (15.0% de los datos).", space_after=6)
    add_p("Esta división fue serializada en data/dataset_splits.json, garantizando que los seis modelos y los dos escenarios de características operen exactamente sobre los mismos conjuntos sin alteración.")

    add_h2("3.5. Pipeline de Segmentación OpenCV y Generación de Máscaras")
    add_p("El módulo hsv_segmenter.py implementa una estrategia modular adaptada a las características radiométricas de las piezas LEGO:")
    add_p("1. Conversión de BGR a HSV: Se descompone la imagen en los planos $H$, $S$ y $V$.", space_after=2)
    add_p("2. Máscara de Piezas Coloreadas ($M_S$): Se aplica el método de umbralización automática de Otsu sobre el canal de Saturación $S$. El fondo translúcido posee $S < 100$, mientras que las piezas plásticas saturadas (azul, rojo, amarillo, verde, marrón) generan picos elevados en el histograma.", space_after=2)
    add_p("3. Máscara de Piezas Oscuras y Negras ($M_V$): Como las piezas negras carecen de saturación alta pero absorben fuertemente la luz retroiluminada ($V < 130$), se calcula un umbral inverso en el canal de Valor.", space_after=2)
    add_p("4. Fusión Lógica OR: Se combinan ambas señales mediante $M_{\\text{comb}} = M_S \\lor M_V$.", space_after=2)
    add_p("5. Operaciones Morfológicas de Limpieza: Se aplica una apertura con elemento estructurante rectangular de $5 \\times 5$ píxeles para suprimir ruido granular perimetral, seguida de un cierre morfológico de $9 \\times 9$ píxeles (2 iteraciones) para sellar los valles de sombra generados entre los tetones.", space_after=2)
    add_p("6. Filtrado por Área Mínima y Detección de Contornos: cv2.findContours() extrae los polígonos externos. Se descartan contornos espurios con $\\text{Área} < 400$ px.", space_after=2)
    add_p("7. Extracción con Margen de Guarda (Padding): Se añade un margen perimetral configurable (margin = 15 px) alrededor del bounding box para evitar el cercenamiento de esquinas inclinadas o bordes exteriores de la pieza.", space_after=6)

    add_h2("3.6. Evaluación Experimental de Watershed")
    add_p("En cumplimiento del numeral 9.4 del proyecto, se evaluó el algoritmo de Watershed como técnica para separar piezas superpuestas o en contacto directo. Utilizando la transformada de distancia Euclidiana $D(x, y) = \\min_{B} \\| (x, y) - B \\|$, se calcularon los marcadores de primer plano seguro ($\text{umbral} = 0.45 \\cdot D_{\\max}$) y la región desconocida de frontera.")
    add_p("Hallazgo Experimental: Se demostró que cuando dos piezas están en contacto, Watershed logra desacoplarlas en dos componentes independientes. Sin embargo, su aplicación indiscriminada sobre piezas individuales con tetones prominentes provoca sobre-segmentación artificial, dividiendo los relieves superiores del cuerpo principal de la pieza. Por ende, Watershed se implementó como un módulo opcional activable únicamente cuando se detecta conectividad múltiple.")

    add_image_if_exists("results/segmentation/watershed_separation_analysis.png", "Figura 3. Evaluación del algoritmo Watershed sobre piezas en contacto versus segmentación estándar.")

    add_h2("3.7. Extracción de Características: Escenario 1 vs. Escenario 2")
    add_p("El extractor feature_extractor.py implementó dos espacios vectoriales normalizados estrictamente con StandardScaler ajustado en el conjunto de entrenamiento:")
    add_p("• Escenario 1 (Solo Forma y Geometría - 301 dimensiones):\n"
          "  - 6 medidas geométricas: área normalizada, perímetro normalizado, relación de aspecto ($w/h$), extensión ($A/wh$), solidez ($A/A_{\\text{hull}}$) y compacidad ($4\\pi A/P^2$).\n"
          "  - 7 Momentos de Hu con transformación logarítmica con signo.\n"
          "  - 288 descriptores HOG calculados en escala de grises sobre recortes redimensionados a $64 \\times 64$ (células de $16\\times 16$, bloques $2\\times 2$, 8 orientaciones).", space_after=4)
    add_p("• Escenario 2 (Forma, Geometría y Color - 325 dimensiones):\n"
          "  - Las 301 características de forma anteriores.\n"
          "  - 24 histogramas normalizados de color HSV (8 bins para H, 8 bins para S, 8 bins para V).", space_after=6)

    # =========================================================================
    # CAPÍTULO 4: RESULTADOS EXPERIMENTALES
    # =========================================================================
    add_h1("CAPÍTULO 4. RESULTADOS EXPERIMENTALES")

    add_h2("4.1. Validación Cuantitativa de la Segmentación OpenCV")
    add_p("La evaluación cuantitativa se ejecutó sobre un conjunto representativo de 35 pares de imágenes base con máscaras de referencia anotadas y extraídas por Template Matching local:")

    add_p("Tabla 2. Métricas cuantitativas de segmentación frente a máscaras de referencia ground-truth.", bold_prefix="Tabla 2: ", space_after=4)
    table_seg = [
        ["Método de Segmentación", "IoU (Jaccard)", "Coeficiente Dice", "Pixel Accuracy", "Precision (Píxel)", "Recall (Píxel)"],
        ["HSV Dual Otsu (Propuesto)", "0.6457", "0.7011", "0.8192", "0.6606", "0.9516"],
        ["HSV Adaptativo + Canny", "0.6373", "0.6965", "0.8192", "0.6466", "0.9624"],
        ["Sustracción de Fondo (BG Ref)", "0.6785", "0.7895", "0.9813", "0.7682", "0.9043"]
    ]
    t_seg = doc.add_table(rows=len(table_seg), cols=len(table_seg[0]))
    t_seg.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table_seg):
        for j, val in enumerate(row):
            cell = t_seg.cell(i, j)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            if i == 0:
                set_cell_shading(cell, "D9E1F2")
                p.runs[0].bold = True
            elif i == 1:
                set_cell_shading(cell, "EBF1F5")
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            
    add_p("Interpretación: El método propuesto HSV Dual Otsu logra un sobresaliente Recall de 95.16%, lo que garantiza que prácticamente la totalidad del cuerpo de la pieza LEGO queda retenida dentro de la máscara binaria sin cercenar sus bordes estructurales.", space_after=8)

    add_image_if_exists("results/segmentation/segmentation_sample_1_Brick_2x2_L.png", "Figura 4. Ejemplo de segmentación: Original, Máscara Ground Truth, Máscara HSV Otsu y Superposición.")

    add_h2("4.2. Análisis Experimental: Impacto de la Forma vs. Color")
    add_p("El entrenamiento de los modelos bajo los dos escenarios de características arrojó resultados concluyentes en el conjunto de prueba independiente:")

    add_p("Tabla 3. Comparación de Exactitud (Accuracy %) entre Solo Forma vs. Forma + Color.", bold_prefix="Tabla 3: ", space_after=4)
    table_scen = [
        ["Modelo Clasificador", "Solo Forma (301 feats)", "Forma + Color (325 feats)", "Diferencia Absoluta", "Aporte Relativo del Color"],
        ["Perceptrón (OvR)", "74.24%", "76.27%", "+2.03%", "Marginal (+2.7%)"],
        ["SGAdaline (SGD MSE)", "66.38%", "69.72%", "+3.34%", "Moderado (+5.0%)"],
        ["SVM (Kernel RBF)", "89.81%", "90.83%", "+1.02%", "Mínimo (+1.1%)"],
        ["Árbol de Decisión (CART)", "65.21%", "67.83%", "+2.62%", "Marginal (+4.0%)"],
        ["Red Neuronal (ANN / MLP)", "88.06%", "89.23%", "+1.17%", "Mínimo (+1.3%)"]
    ]
    t_sc = doc.add_table(rows=len(table_scen), cols=len(table_scen[0]))
    t_sc.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table_scen):
        for j, val in enumerate(row):
            cell = t_sc.cell(i, j)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            if i == 0:
                set_cell_shading(cell, "D9E1F2")
                p.runs[0].bold = True
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)

    add_image_if_exists("results/models/shape_vs_color_comparison.png", "Figura 5. Comparación cuantitativa del rendimiento: Solo Forma vs. Forma + Color en los 5 modelos tradicionales.")

    add_h2("4.3. Tabla Comparativa General de los Seis Modelos")
    add_p("La evaluación comparativa sobre el conjunto de prueba independiente de 687 muestras con las 20 categorías presenta las siguientes métricas definitivas:")

    add_p("Tabla 4. Tabla comparativa general de rendimiento, tiempos de entrenamiento y latencia de inferencia.", bold_prefix="Tabla 4: ", space_after=4)
    table_bench = [
        ["Modelo", "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", "F1-Score (Weighted)", "Tiempo Entrenamiento", "Latencia Inferencia"],
        ["CNN (Convolucional PyTorch)", "95.63%", "95.63%", "95.58%", "95.55%", "95.63%", "2434.37 s (~40.5 min)", "11.94 ms/muestra"],
        ["SVM (Kernel RBF, C=5.0)", "90.83%", "91.46%", "90.49%", "90.77%", "90.79%", "12.01 s", "1.27 ms/muestra"],
        ["Red Neuronal (ANN / MLP)", "89.23%", "89.19%", "89.03%", "88.99%", "89.13%", "10.51 s", "0.009 ms/muestra"],
        ["Perceptrón (One-vs-Rest)", "76.27%", "76.59%", "76.12%", "76.06%", "76.12%", "1.89 s", "0.026 ms/muestra"],
        ["SGAdaline (SGD MSE Multiclase)", "69.72%", "70.53%", "68.86%", "68.68%", "69.22%", "45.46 s", "0.030 ms/muestra"],
        ["Árbol de Decisión (CART Gini)", "67.83%", "68.11%", "67.02%", "67.02%", "67.97%", "1.48 s", "0.001 ms/muestra"]
    ]
    t_b = doc.add_table(rows=len(table_bench), cols=len(table_bench[0]))
    t_b.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table_bench):
        for j, val in enumerate(row):
            cell = t_b.cell(i, j)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            if i == 0:
                set_cell_shading(cell, "1F4E79")
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
                p.runs[0].bold = True
            elif i == 1:
                set_cell_shading(cell, "E2EFDA") # Winner green
            elif i == 2:
                set_cell_shading(cell, "FFF2CC") # Efficiency winner yellow
            set_cell_margins(cell, top=60, bottom=60, left=60, right=60)

    add_image_if_exists("results/models/models_benchmark_comparison.png", "Figura 6. Gráficas comparativas de rendimiento (Accuracy y F1-Score), tiempo de entrenamiento y latencia de inferencia.")

    add_h2("4.4. Análisis Individual de Modelos y Curvas de Aprendizaje")
    
    add_h3("A. Red Neuronal Convolucional (CNN) - Modelo Ganador Absoluto")
    add_p("La arquitectura convolucional (3 bloques de convolución doble $3\\times 3$ con BatchNorm2d, MaxPool2d, AdaptiveAvgPool2d y cabezal denso de 256 neuronas con Dropout 0.4) alcanzó una convergencia excelente en 25 épocas con el optimizador AdamW y Cosine Annealing. Alcanzó 95.63% de Exactitud, superando por más de 4.8 puntos porcentuales al mejor modelo tradicional.")
    add_image_if_exists("results/models/cnn_learning_curves.png", "Figura 7. Curvas de aprendizaje de la CNN: Pérdida Cross-Entropy y Exactitud de Validación por época.")
    add_image_if_exists("results/models/cm_cnn.png", "Figura 8. Matriz de confusión normalizada de la Red Neuronal Convolucional (CNN).")

    add_h3("B. Support Vector Machine (SVM) - Modelo Óptimo en Eficiencia")
    add_p("La exploración sistemática de hiperparámetros en el conjunto de validación confirmó la superioridad del kernel RBF con $C=5.0$ (89.96% en validación, 90.83% en prueba) frente al kernel lineal (83.70%) y polinómico de grado 3 (88.21%). Logra un rendimiento cercano a la CNN con una reducción del 99.5% en tiempo de entrenamiento (12 segundos vs 40 minutos).")
    add_image_if_exists("results/models/cm_svm.png", "Figura 9. Matriz de confusión normalizada del modelo SVM (Kernel RBF).")

    add_h3("C. Red Neuronal Artificial (ANN / MLP)")
    add_p("La red densa multicapa ($325 \\to 256 \\to 128 \\to 20$) entrenada durante 35 épocas alcanzó 89.23% de Exactitud y 88.99% de F1-Macro. La incorporación de Batch Normalization y Dropout evitó el sobreajuste, logrando la segunda latencia de inferencia más veloz (9 microsegundos por muestra).")
    add_image_if_exists("results/models/ann_learning_curves.png", "Figura 10. Curvas de aprendizaje de la ANN (MLP): Pérdida y Exactitud de validación.")

    add_h3("D. Modelos Lineales: Perceptrón y SGAdaline")
    add_p("El Perceptrón alcanzó 76.27% de Exactitud, mientras que SGAdaline obtuvo 69.72%. Ambos algoritmos evidencian la limitación teórica de separar 20 clases en un espacio geométricamente entrelazado mediante hiperplanos estrictamente lineales. La función de pérdida cuadrática MSE de Adaline mostró una sensibilidad mayor a la dispersión de los histogramas de color en comparación con la pérdida Hinge de SVM.")

    add_h3("E. Árboles de Decisión (CART)")
    add_p("El árbol optimizado con criterio Gini y profundidad máxima de 14 alcanzó 67.83% de Exactitud. El análisis de importancia de características reveló que las variables geométricas (extensión, relación de aspecto y solidez) y las celdas centrales del descriptor HOG concentran más del 65% del poder de división del árbol.")
    add_image_if_exists("results/models/decision_tree_feature_importance.png", "Figura 11. Top 20 características con mayor importancia relativa en el Árbol de Decisión.")

    # =========================================================================
    # CAPÍTULO 5: DISCUSIÓN
    # =========================================================================
    add_h1("CAPÍTULO 5. DISCUSIÓN")
    add_p("El análisis transversal de los resultados empíricos permite responder con solidez científica a las interrogantes planteadas:")
    add_p("1. ¿Por qué la CNN superó a los modelos basados en características manuales? Las características diseñadas artesanalmente (Hu, HOG, geometría) colapsan la imagen a un vector 1D perdiendo la disposición jerárquica de los tetones cilíndricos y las pendientes sutiles de los bloques Slope. La CNN conserva la topología 2D en cada mapa de activación convolucional, aprendiendo relaciones espaciales invariantes.", space_after=4)
    add_p("2. ¿Qué influencia real tiene el color en el reconocimiento de piezas LEGO? La comparación directa entre Escenario 1 y Escenario 2 demostró que la geometría aporta entre el 88% y el 90% de la capacidad discriminativa. El color añade entre 1.0% y 3.3% de ganancia estadística. Si bien ayuda ligeramente a separar ciertas piezas oscuras del fondo, la identidad de una pieza LEGO reside enteramente en su forma.", space_after=4)
    add_p("3. ¿Cuáles son las categorías de mayor dificultad? Las matrices de confusión revelan que los errores más frecuentes ocurren entre piezas de igual grosor y longitud contigua (por ejemplo, Plate 1x2 confundido con Plate 1x3, o Plate 1x4 con Plate 2x4). Asimismo, en ángulos cenitales planos (perpendiculares a $90^\circ$), distinguir un Plate de un Brick es sumamente complejo debido a que la altura tridimensional no se proyecta.", space_after=4)
    add_p("4. ¿Qué limitaciones presenta la segmentación HSV tradicional ante fotografías del mundo real? Como se observó en las pruebas con imágenes externas (como fotografías tomadas sosteniendo piezas con la mano), el algoritmo HSV no posee semántica de objeto: clasifica la piel humana como primer plano debido a su saturación natural ($S \\approx 50-120$). Para corregir esto, se incorporó un filtro de piel en el espacio YCrCb ($Cr \\in [133, 173], Cb \\in [77, 127]$) y un umbral de confianza mínima, aislando la pieza plástica.", space_after=4)
    add_p("5. ¿Cuál es el compromiso óptimo entre precisión y costo computacional? Aunque la CNN es la ganadora en exactitud (95.63%), requirió 40.5 minutos de entrenamiento en CPU. Por el contrario, SVM RBF alcanzó 90.83% de exactitud entrenándose en solo 12 segundos, convirtiéndose en la opción idónea para sistemas embebidos de baja potencia.", space_after=6)

    # =========================================================================
    # CAPÍTULO 6: INTEGRACIÓN DEL SISTEMA
    # =========================================================================
    add_h1("CAPÍTULO 6. INTEGRACIÓN DEL SISTEMA Y PIPELINE END-TO-END")
    add_p("El flujo funcional implementado en pipeline.py articula de extremo a extremo las etapas del sistema mediante los siguientes 8 pasos secuenciales:")
    add_p("Paso 1: Recepción de la fotografía de entrada (resolución nativa BGR).\n"
          "Paso 2: Preprocesamiento y filtrado de piel opcional en espacio YCrCb para suprimir dedos/manos.\n"
          "Paso 3: Segmentación por color en espacio HSV aplicando umbralización dual Otsu ($M_S \\lor M_V$).\n"
          "Paso 4: Limpieza morfológica (apertura $5\\times 5$, cierre $9\\times 9$) y filtrado de área mínima ($>400$ px).\n"
          "Paso 5: Detección de contornos externos y extracción de cajas delimitadoras con margen perimetral (+15 px).\n"
          "Paso 6: Recorte de cada región de interés (ROI) y redimensionamiento con interpolación anti-aliasing a $64 \\times 64$.\n"
          "Paso 7: Inferencia y cálculo de probabilidades Softmax mediante la Red Neuronal Convolucional (CNN).\n"
          "Paso 8: Generación y almacenamiento del reporte visual cuádruple: (1) Fotografía Original, (2) Máscara Binaria, (3) Pieza Aislada y (4) Superposición con Contorno, Bounding Box, Etiqueta de Clase y Confianza (%).", space_after=6)

    add_image_if_exists("results/predictions/detection_2_Brick_2x2_L_180708220529_hsv_otsu.png", "Figura 12. Salida del pipeline integrado: panel cuádruple con detección y clasificación de Brick_2x2_L al 98.0% de confianza.")

    # =========================================================================
    # CAPÍTULO 7: CONCLUSIONES
    # =========================================================================
    add_h1("CAPÍTULO 7. CONCLUSIONES")
    add_p("1. Se desarrolló e implementó con éxito un sistema modular de visión artificial que resuelve de manera integral las tareas de segmentación, extracción morfológica y clasificación multiclase de piezas LEGO sobre 20 categorías.", space_after=3)
    add_p("2. El algoritmo de segmentación en OpenCV en espacio HSV combinando Otsu en saturación y valor inverso demostró ser altamente efectivo en escenas controladas, alcanzando un Recall de 95.16% y Pixel Accuracy de 81.92%, garantizando la preservación íntegra de las fronteras físicas de los bloques.", space_after=3)
    add_p("3. La experimentación comparativa demostró de forma concluyente que la Red Neuronal Convolucional (CNN) es el modelo con mejor desempeño general, alcanzando 95.63% de Exactitud y 95.55% de F1-Score Macro, demostrando la superioridad del aprendizaje automático de características visuales frente a descriptores diseñados a mano.", space_after=3)
    add_p("4. La Support Vector Machine (SVM con kernel RBF) demostró ser el modelo de mayor eficiencia relativa, alcanzando 90.83% de Exactitud con un tiempo de entrenamiento de apenas 12.01 segundos y latencia de inferencia de 1.27 ms por muestra.", space_after=3)
    add_p("5. El análisis empírico entre los escenarios de representación confirmó que la geometría y los gradientes espaciales de los tetones (HOG y Hu) explican cerca del 90% del poder predictivo, mientras que el color aporta un complemento menor que debe desacoplarse para evitar sesgos cromáticos.", space_after=3)
    add_p("6. La inclusión de un filtro de supresión de piel en espacio YCrCb resolvió eficazmente la segmentación en fotografías no controladas donde las piezas son sostenidas con los dedos, permitiendo la interacción natural en la aplicación web Streamlit.", space_after=6)

    # =========================================================================
    # CAPÍTULO 8: TRABAJO FUTURO
    # =========================================================================
    add_h1("CAPÍTULO 8. TRABAJO FUTURO")
    add_p("A partir de las lecciones aprendidas y los desafíos abiertos, se identifican las siguientes líneas de desarrollo futuro:")
    add_p("• Segmentación de Instancias basada en Deep Learning: Reemplazar el segmentador morfológico clásico de OpenCV por redes neuronales convolucionales de segmentación como Mask R-CNN o YOLOv8-seg, capaces de aislar piezas sobre fondos altamente texturados (alfombras, madera, manteles) sin requerir contrastes cromáticos ideales.", space_after=3)
    add_p("• Estimación de Pose Tridimensional (6D Pose Estimation): Integrar redes de estimación de pose como DenseFusion o PVNet para calcular la rotación y traslación tridimensional exacta de la pieza en el espacio cartesiano, facilitando su integración en brazos robóticos industriales de pick-and-place.", space_after=3)
    add_p("• Transfer Learning con Arquitecturas Preentrenadas: Evaluar redes preentrenadas en ImageNet mediante fine-tuning (ResNet-18, MobileNetV3-Small o EfficientNet-B0), optimizadas mediante cuantización INT8 y exportación a TensorRT / ONNX para reducir la latencia de inferencia a menos de 2 milisegundos en procesadores embebidos.", space_after=3)
    add_p("• Expansión del Catálogo y Reconocimiento Open-Set: Implementar arquitecturas de aprendizaje métrico (Siamese Networks o pérdidas de contraste como ArcFace / SupCon) para permitir la identificación de piezas LEGO raras no vistas durante el entrenamiento mediante distancias en el espacio latente de incrustación.", space_after=6)

    # =========================================================================
    # REFERENCIAS BIBLIOGRÁFICAS (APA 7ma Edición)
    # =========================================================================
    add_h1("REFERENCIAS BIBLIOGRÁFICAS")
    
    refs = [
        "Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees. CRC Press.",
        "Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools, 25(11), 120-125.",
        "Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297. https://doi.org/10.1007/BF00994018",
        "Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems, 2(4), 303-314. https://doi.org/10.1007/BF02551274",
        "Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. En 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05) (Vol. 1, pp. 886-893). IEEE. https://doi.org/10.1109/CVPR.2005.177",
        "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. En Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).",
        "Hu, M. K. (1962). Visual pattern recognition by moment invariants. IRE Transactions on Information Theory, 8(2), 179-187. https://doi.org/10.1109/TIT.1962.1057692",
        "LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324. https://doi.org/10.1109/5.726791",
        "Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66. https://doi.org/10.1109/TSMC.1979.4310076",
        "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32, 8026-8037.",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
        "Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65(6), 386-408. https://doi.org/10.1037/h0042519",
        "Suzuki, S., & Abe, K. (1985). Topological structural analysis of digitized binary images by border following. Computer Vision, Graphics, and Image Processing, 30(1), 32-46. https://doi.org/10.1016/0734-189X(85)90016-7",
        "van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., ... & scikit-image contributors. (2014). scikit-image: Image processing in Python. PeerJ, 2, e453. https://doi.org/10.7717/peerj.453",
        "Widrow, B., & Hoff, M. E. (1960). Adaptive switching circuits. IRE WESCON Convention Record, 4, 96-104."
    ]
    for r in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.5)
        p_ref.paragraph_format.first_line_indent = Inches(-0.5)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.add_run(r)

    # Save document
    doc.save(DOCX_OUT)
    print(f"Documento Word (.docx) generado exitosamente en: {DOCX_OUT}")

if __name__ == "__main__":
    create_document()
