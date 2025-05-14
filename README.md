# app.py
correción de código con IA
# -*- coding: utf-8 -*-
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pygments.lexers import guess_lexer, get_lexer_by_name
from pygments.util import ClassNotFound
import re

from PIL import Image

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente IA - Corrección y Sugerencia",
    layout="wide",
)

# Carga de la imagen del sello
sello = Image.open("img.jpg")

# Crear dos columnas: una para la imagen y otra para el título
col1, col2 = st.columns([1, 3])  # Ajusta proporciones según el tamaño de la imagen

with col1:
    sello = sello.resize((220, 250))  # (ancho, alto)
    st.image(sello)

with col2:
    with col2:
        st.markdown(
      """
        <div style='text-align: center; line-height: 1.0;'>
            <h1 style='margin-bottom: 5px;'>
                UNIDAD EDUCATIVA FISCAL<br><span style="font-size: 90%;">"GRAN BRETAÑA"</span>
            </h1>
            <h3>🤖 Asistente IA - Corrección y Sugerencia de código</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Carga del Modelo ---
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small")
        model = AutoModelForCausalLM.from_pretrained("codeparrot/codeparrot-small")
        return tokenizer, model, True, None
    except Exception as e:
        return None, None, False, e

tokenizer, model, model_loaded, load_error = load_model()

if not model_loaded:
    st.error(f"Error al cargar el modelo de IA: {load_error}")
    st.warning("La función de sugerencia de IA no estará disponible.")

# --- Detección de Lenguaje (Sin cambios grandes aquí) ---
def detect_language(codigo):
    # ... (mantenemos la versión anterior de detect_language) ...
    if not codigo or not codigo.strip(): return "Unknown"
    stripped_code = codigo.strip()
    lower_code = stripped_code.lower()
    # 1. Reglas Regex Prioritarias
    if re.search(r'#include\s*<.*?>', stripped_code): return "C"
    if re.search(r'import\s+java\..*;', stripped_code) or re.search(r'public\s+class\s+\w+', stripped_code) : return "Java"
    if re.search(r'^\s*(def|class)\s+\w+\(.*\):', codigo, re.MULTILINE): return "Python"
    if re.search(r'import\s+\w+', stripped_code) and not re.search(r'java\.', stripped_code): return "Python"
    # Heurística print/printf
    if 'printn(' in lower_code or 'print(' in lower_code:
        if re.search(r':\s*$', codigo, re.MULTILINE): return "Python"
        if 'printf(' not in lower_code: return "Python" # Suposición
    if 'printf(' in lower_code: return "C" # Suposición
    # 2. Pygments
    try:
        lexer = guess_lexer(stripped_code)
        generic_lexers = {"text", "textonly", "special", "fundamental"}
        lexer_name_lower = lexer.name.lower()
        if lexer.name and lexer_name_lower not in generic_lexers:
             if lexer_name_lower == "python 3": return "Python"
             if lexer_name_lower == "c": return "C"
             if lexer_name_lower == "java": return "Java"
             return lexer.name
    except ClassNotFound: pass
    except Exception as e: print(f"Error inesperado en guess_lexer: {e}")
    # 3. Heurísticas Adicionales
    if re.search(r'\):$', stripped_code, re.MULTILINE):
         if re.search(r'\b(if|for|while|def|class)\b', stripped_code): return "Python"
    # 4. Fallback
    return "Unknown"


# --- Revisión de Sintaxis Simple ---
def revisar_sintaxis_simple(codigo):
    """Busca errores muy básicos como paréntesis o comillas sin cerrar."""
    errores = []
    # Conteo simple de paréntesis (no anidado perfecto, pero ayuda)
    if codigo.count('(') > codigo.count(')'):
        errores.append("Parece que falta un paréntesis de cierre ')'.")
    elif codigo.count(')') > codigo.count('('):
         errores.append("Parece que sobra un paréntesis de cierre ')'.")
    # Conteo simple de llaves
    if codigo.count('{') > codigo.count('}'):
        errores.append("Parece que falta una llave de cierre '}'.")
    elif codigo.count('}') > codigo.count('{'):
         errores.append("Parece que sobra una llave de cierre '}'.")
    # Conteo simple de corchetes
    if codigo.count('[') > codigo.count(']'):
        errores.append("Parece que falta un corchete de cierre ']'.")
    elif codigo.count(']') > codigo.count('['):
        errores.append("Parece que sobra un corchete de cierre ']'.")

    # Buscar comillas sin pareja (más complejo por strings dentro de strings, pero intentamos algo básico)
    # Contar comillas dobles y simples. Si es impar, probablemente falte una.
    if codigo.count('"') % 2 != 0:
        errores.append('Parece que falta una comilla doble " para cerrar un texto.')
    if codigo.count("'") % 2 != 0:
        errores.append("Parece que falta una comilla simple ' para cerrar un texto.")

    return errores


# --- Sugerencia IA (con revisión previa) ---
def sugerencia_ia(codigo):
    """Genera sugerencia, pero antes revisa errores básicos."""
    if not model_loaded: return "# Modelo de IA no cargado."
    if not codigo or len(codigo.strip()) < 5: # Aumentado un poco el mínimo
        return "# El código es muy corto o está vacío para generar una sugerencia útil."

    # *** NUEVA REVISIÓN DE SINTAXIS ***
    errores_sintaxis = revisar_sintaxis_simple(codigo)
    if errores_sintaxis:
        mensaje_error = "# Tu código parece tener errores básicos:\n"
        for err in errores_sintaxis:
            mensaje_error += f"# - {err}\n"
        mensaje_error += "# La IA podría dar resultados inesperados. Intenta corregir tu código primero."
        return mensaje_error

    # Si no hay errores obvios, proceder con la IA
    try:
        max_input_length = 512
        codigo_truncado = codigo[-max_input_length:]
        inputs = tokenizer.encode(codigo_truncado, return_tensors="pt")
        max_new = 40
        outputs = model.generate(
            inputs, max_length=inputs.shape[1] + max_new, num_return_sequences=1,
            do_sample=True, temperature=0.7, top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
        sugerencia = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        sugerencia_limpia = "\n".join(sugerencia.strip().split('\n')).strip()

        if not sugerencia_limpia or len(sugerencia_limpia) < 5:
             return "# No se pudo generar una sugerencia útil con este código."

        return sugerencia_limpia

    except Exception as e:
        st.error(f"Error durante la generación de IA: {e}")
        return f"# Error al generar sugerencia: {e}"

# --- Corrección Basada en Reglas (sin cambios internos) ---
def corregir_codigo(codigo, lenguaje):
    # ... (mantenemos la versión anterior de corregir_codigo) ...
    if not codigo or lenguaje == "Unknown": return codigo, None, "Lenguaje desconocido, no se aplicaron correcciones."
    lineas = codigo.split('\n')
    corregidas = []
    cambios_realizados = []
    mensaje_error = None
    keywords_control_flujo_generales = {'if', 'for', 'while', 'switch', 'else', 'try', 'catch', 'finally', 'class', 'interface', 'enum'}
    keywords_definicion = {'void', 'int', 'float', 'double', 'char', 'boolean', 'def', 'class', 'struct', 'enum', 'interface'}
    for i, linea in enumerate(lineas):
        original = linea; linea_trabajo = linea; linea_strip = linea.strip()
        if not linea_strip: corregidas.append(linea_trabajo); continue
        if lenguaje == "Python":
            if linea_strip.startswith("//"): linea_trabajo = linea_trabajo.replace("//", "#", 1)
            elif linea_strip.startswith("#"): corregidas.append(linea_trabajo); continue
            if "printf(" in linea_trabajo: linea_trabajo = linea_trabajo.replace("printf(", "print(")
            elif "printn(" in linea_trabajo: linea_trabajo = linea_trabajo.replace("printn(", "print(")
            match_bloque = re.match(r"^\s*(def|if|for|while|else|elif|try|except|finally|with|class)\s+.*", linea_strip)
            if match_bloque and not linea_strip.endswith(":") and not linea_strip.startswith("#"):
                 indentacion = linea_trabajo[:len(linea_trabajo) - len(linea_trabajo.lstrip())]
                 linea_trabajo = indentacion + linea_strip + ":"
        elif lenguaje == "C":
            if linea_strip.startswith("#") and not linea_strip.startswith(("#include", "#define", "#if", "#ifdef", "#ifndef", "#else", "#endif")): linea_trabajo = linea_trabajo.replace("#", "//", 1)
            elif linea_strip.startswith(("//", "/*")): corregidas.append(linea_trabajo); continue
            if "print(" in linea_trabajo and "printf(" not in linea_trabajo: linea_trabajo = linea_trabajo.replace("print(", "printf(")
            elif "printn(" in linea_trabajo: linea_trabajo = linea_trabajo.replace("printn(", "printf(")
            palabra_inicial = linea_strip.split(' ')[0].split('(')[0] if linea_strip else ''
            if linea_strip.endswith(')') and not linea_strip.endswith((';', '{', ',')) and not palabra_inicial in keywords_control_flujo_generales | keywords_definicion and not linea_strip.startswith(('#', '//', '/*')) :
                 if not (palabra_inicial in keywords_definicion and '(' in linea_strip and ')' in linea_strip and '{' not in linea_strip): linea_trabajo += ";"
        elif lenguaje == "Java":
            if linea_strip.startswith("#"): linea_trabajo = linea_trabajo.replace("#", "//", 1)
            elif linea_strip.startswith(("//", "/*")): corregidas.append(linea_trabajo); continue
            match_print = re.match(r"(\s*)(printn?)\((.*)\)(\s*;?)(\s*)", linea_trabajo)
            if match_print:
                 indent, _, contenido, punto_coma_existente, trailing = match_print.groups()
                 linea_trabajo = f"{indent}System.out.println({contenido});{trailing if trailing else ''}"
            palabra_inicial = linea_strip.split(' ')[0].split('(')[0] if linea_strip else ''
            if linea_strip and not linea_strip.endswith(('{', '}', ';', ',')) and not linea_strip.startswith(('/', '#')) and palabra_inicial not in keywords_control_flujo_generales and not re.search(r'\b(class|interface|enum)\b', linea_strip) and not re.match(r".*\)\s*\{$", linea_strip):
                   if linea_strip.endswith(')') or (re.search(r'\s+=\s+', linea_strip) and '(' not in linea_strip.split('=')[0]):
                         if not (palabra_inicial in keywords_definicion and '(' in linea_strip and ')' in linea_strip and '{' not in linea_strip): linea_trabajo += ";"
        if linea_trabajo != original: cambios_realizados.append(f"🔄 Línea {i+1}: `{original.strip()}` → `{linea_trabajo.strip()}`")
        corregidas.append(linea_trabajo)
    codigo_corregido = '\n'.join(corregidas)
    mensaje_cambios = "\n".join(cambios_realizados) if cambios_realizados else None
    if not mensaje_cambios and lenguaje != "Unknown": mensaje_error = f"No se encontraron errores simples para corregir en {lenguaje} con las reglas actuales."
    return codigo_corregido, mensaje_cambios, mensaje_error

# --- Interfaz Streamlit (con revisión de sugerencia) ---


st.info("Hola Usuario 😊 Pega tu código, intentaré detectar el lenguaje y podrás corregir errores *simples* o pedir una sugerencia a la IA.")
st.info(
    "ℹ️ **Importante:**\n"
    "- La corrección busca errores comunes "
)

# Inicialización del estado
if "codigo_usuario" not in st.session_state:
    st.session_state.codigo_usuario = """printf("hola); // <-- ¡Error! Falta " ) ;"""
if "resultado_mostrado" not in st.session_state:
     st.session_state.resultado_mostrado = st.session_state.codigo_usuario

col_in, col_out = st.columns(2)

with col_in:
    st.subheader("✍️ Tu Código:")
    codigo_input = st.text_area(
        "Introduce o pega tu código aquí:",
        value=st.session_state.codigo_usuario, height=400, key="codigo_input_area",
        on_change=lambda: st.session_state.update(resultado_mostrado=st.session_state.codigo_input_area)
    )
    st.session_state.codigo_usuario = codigo_input
    lenguaje_detectado_actual = detect_language(codigo_input)
    lang_display = lenguaje_detectado_actual if lenguaje_detectado_actual != "Unknown" else "Desconocido"
    st.caption(f"Lenguaje detectado (aproximado): **{lang_display}**")

    # Mostrar advertencias de sintaxis simple directamente aquí
    errores_sint_input = revisar_sintaxis_simple(codigo_input)
    if errores_sint_input:
        st.warning("⚠️ ¡Ojo! Tu código parece tener errores:", icon="❗")
        for err in errores_sint_input:
            st.markdown(f"- {err}")


    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1: corregir_btn = st.button("🛠️ Corregir Errores Simples", use_container_width=True)
    with b_col2: sugerencia_btn = st.button("💡 Sugerencia IA", use_container_width=True, disabled=not model_loaded)
    with b_col3: limpiar_btn = st.button("🧹 Limpiar Todo", use_container_width=True)

with col_out:
    st.subheader("✅ Resultado / Sugerencia:")
    output_placeholder = st.empty()

# --- Lógica de los Botones ---

if limpiar_btn:
    st.session_state.codigo_usuario = ""
    st.session_state.resultado_mostrado = ""
    st.rerun()

if corregir_btn:
    codigo_actual = st.session_state.codigo_usuario
    if codigo_actual.strip():
        lenguaje = detect_language(codigo_actual)
        corregido, cambios, msg_error = corregir_codigo(codigo_actual, lenguaje)
        st.session_state.resultado_mostrado = corregido
        with output_placeholder.container():
            lang_pygments = lenguaje.lower() if lenguaje != "Unknown" else "text"
            try: get_lexer_by_name(lang_pygments)
            except ClassNotFound: lang_pygments = "text"
            st.code(corregido, language=lang_pygments, line_numbers=True)
            if cambios:
                st.success("Posibles correcciones aplicadas:")
                st.markdown(cambios)
            elif msg_error:
                 st.info(msg_error)
    else:
        st.warning("Introduce código para corregir.")
        st.session_state.resultado_mostrado = ""
        output_placeholder.text("")

if sugerencia_btn:
    codigo_actual = st.session_state.codigo_usuario
    if not model_loaded: st.error("Modelo IA no cargado.")
    elif codigo_actual.strip():
        with st.spinner('Pensando... 🤔'):
            sugerencia = sugerencia_ia(codigo_actual) # Ya incluye la revisión de sintaxis

        # Mostrar Código Original y Sugerencia/Advertencia
        st.session_state.resultado_mostrado = codigo_actual # Base para mostrar
        with output_placeholder.container():
            lang_original = detect_language(codigo_actual).lower()
            try: get_lexer_by_name(lang_original)
            except ClassNotFound: lang_original = "text"

            st.markdown("#### Código Original:")
            st.code(codigo_actual, language=lang_original, line_numbers=True)
            st.markdown("#### 💡 Sugerencia de la IA:")

            # Si la sugerencia es una advertencia de error de sintaxis o similar
            if sugerencia.startswith("#"):
                 st.warning(sugerencia) # Mostrar el mensaje de advertencia/error de la función
            else:
                 # *** NUEVO: Revisar si la sugerencia es del mismo lenguaje ***
                 lang_sugerencia = detect_language(sugerencia).lower()
                 lang_orig_lower = detect_language(codigo_actual).lower() # Detectar de nuevo por si acaso

                 # Comprobar si el lenguaje de la sugerencia coincide o es texto genérico
                 # Permitimos 'unknown' o 'text' en la sugerencia por si es corta
                 if lang_sugerencia == lang_orig_lower or lang_sugerencia in ['unknown', 'text']:
                     st.markdown("> *(Revisa siempre la sugerencia. Es generada automáticamente.)*")
                     st.code(sugerencia, language=lang_original) # Resaltar como el original
                 else:
                     # ¡La sugerencia parece de otro lenguaje!
                     st.warning(f"⚠️ ¡Cuidado! La sugerencia parece ser **{lang_sugerencia.capitalize()}** y tu código original era **{lang_orig_lower.capitalize()}**. La IA podría estar confundida.")
                     st.code(sugerencia, language=lang_sugerencia) # Mostrar con su propio lenguaje detectado

    else:
        st.warning("Introduce código para obtener sugerencia.")
        st.session_state.resultado_mostrado = ""
        output_placeholder.text("")


# Mostrar el resultado/código actual si no se hizo otra cosa
# --- Lógica Final de Visualización ---
# Esta sección se ejecuta después de la lógica de los botones.
# Solo queremos actualizar el placeholder si NINGÚN botón lo hizo ya.

# Verificamos si alguno de los botones principales fue presionado EN ESTA EJECUCIÓN específica.
# Las variables (corregir_btn, sugerencia_btn, limpiar_btn) serán True solo durante
# la ejecución que fue disparada por ese botón específico.
button_action_taken = corregir_btn or sugerencia_btn or limpiar_btn

# Si NINGÚN botón tomó acción en esta ejecución Y hay algo que mostrar en el estado:
if not button_action_taken and "resultado_mostrado" in st.session_state and st.session_state.resultado_mostrado:
    # Entonces, mostramos el contenido del estado 'resultado_mostrado' como vista previa por defecto.
    # Esto maneja casos como cambiar el texto en el input sin pulsar botones.
    try:
        current_lang = detect_language(st.session_state.resultado_mostrado).lower()
        if current_lang == 'unknown': current_lang = 'text'
        try: get_lexer_by_name(current_lang)
        except ClassNotFound: current_lang = "text"

        # Actualizar el placeholder con el estado actual
        output_placeholder.code(st.session_state.resultado_mostrado, language=current_lang, line_numbers=True)

    except Exception as display_error:
        print(f"Error al mostrar estado por defecto: {display_error}")
        # Podrías mostrar un mensaje de error aquí si falla la visualización por defecto
        # output_placeholder.error("Error al mostrar el código.")


# Nota: Si un botón SÍ tomó acción (button_action_taken es True),
# el contenido que ese botón puso en output_placeholder.container() se mantendrá
# y esta sección final simplemente no hará nada, evitando la sobreescritura.
