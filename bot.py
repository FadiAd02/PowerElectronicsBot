import os
import json
from huggingface_hub import InferenceClient
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import rectifier_functionstwo
from huggingface_hub import HfFolder

# Hugging Face API token


from huggingface_hub import login
login(token="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")

huggingface_api_token = HfFolder.get_token()


if not huggingface_api_token:
    raise ValueError("No Hugging Face token found. Log in with 'huggingface-cli login' or set HUGGINGFACE_API_TOKEN environment variable.")

client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=huggingface_api_token)

# Telegram bot token
TELEGRAM_BOT_TOKEN = "7801211720:AAGT8uy67sSFA2Pa-1LWNeEXCMbA-5aUGUY"

# Define FUNCTION_MAP
FUNCTION_MAP = {
    "Half-wave rectifier with resistive load": rectifier_functionstwo.half_wave_rectifier_with_resistive_load,
    "Half-wave rectifier with transformer and resistive load": rectifier_functionstwo.half_wave_rectifier_with_transformer_and_resistive_load,
    "Half-wave rectifier with RL load (standard outputs)": rectifier_functionstwo.half_wave_rectifier_with_rl_load,
    "Half-wave rectifier with RL load (PSpice)": rectifier_functionstwo.half_wave_rectifier_with_rl_load_and_pspice,
    "Half-wave rectifier with RL source load": rectifier_functionstwo.half_wave_rectifier_with_rl_source_load,
    "Half-wave rectifier with L and DC source load": rectifier_functionstwo.half_wave_rectifier_with_l_and_dc_source_load,
    "Half-wave rectifier with freewheeling diode and RL load": rectifier_functionstwo.half_wave_rectifier_with_freewheeling_diode_and_rl_load,
    "Half-wave rectifier with freewheeling diode and RL load (PSpice)": rectifier_functionstwo.half_wave_rectifier_with_freewheeling_diode_and_rl_load_pspice,
    "Half-wave rectifier with freewheeling diode and DC source load": rectifier_functionstwo.half_wave_rectifier_with_freewheeling_diode_and_dc_source_load,
    "Half-wave rectifier with capacitor filter (single resistance)": rectifier_functionstwo.half_wave_rectifier_with_capacitor_filter_single_resistance,
    "Half-wave rectifier with capacitor filter (dual-resistance analysis)": rectifier_functionstwo.half_wave_rectifier_with_capacitor_filter_dual_resistance,
    "Half-wave rectifier with dual capacitor filter": rectifier_functionstwo.half_wave_rectifier_with_dual_capacitor_filter,
    "Half-wave rectifier with capacitor filter (ripple & diode currents)": rectifier_functionstwo.half_wave_rectifier_with_capacitor_filter_ripple_and_diode_currents,
    "Half-wave rectifier with capacitor filter (ripple & diode currents and load power provided)": rectifier_functionstwo.half_wave_rectifier_with_capacitor_filter_ripple_and_diode_currents_P_load,
    "Controlled half-wave rectifier with resistive load": rectifier_functionstwo.controlled_half_wave_rectifier_resistive_load,
    "Controlled half-wave rectifier with resistive load (delay-angle required)": rectifier_functionstwo.controlled_half_wave_rectifier_with_resistive_load_delay,
    "Controlled half-wave rectifier with RL load": rectifier_functionstwo.controlled_half_wave_rectifier_with_rl_load,
    "Half-wave rectifier with RL source load": rectifier_functionstwo.half_wave_rectifier_with_rl_source_load,
    "Controlled half-wave rectifier with RLDC load": rectifier_functionstwo.controlled_half_wave_rectifier_with_rldc_load,
"Controlled half-wave rectifier with series L and DC source (LDC)": rectifier_functionstwo.controlled_half_wave_rectifier_with_ldc,
    "Single-phase full-wave bridge rectifier with resistive load": rectifier_functionstwo.full_wave_bridge_rectifier_resistive_load,
    "Single-phase rectifier comparison": rectifier_functionstwo.single_phase_rectifier_bridge_vs_center_tapped,
    "Single-phase bridge rectifier with RL load (standard outputs)": rectifier_functionstwo.single_phase_bridge_rectifier_with_rl_load,
    "Single-phase bridge rectifier with RL load (power & power factor in the outputs)": rectifier_functionstwo.full_wave_bridge_rectifier_rl_power,
"Full wave bridge rectifier with RL load (power & power factor in the outputs)": rectifier_functionstwo.full_wave_bridge_rectifier_rl_power,
"Full wave rectifier with RL load (power & power factor in the outputs)": rectifier_functionstwo.full_wave_bridge_rectifier_rl_power,
    "Single-phase center-tapped transformer rectifier with resistive load": rectifier_functionstwo.single_phase_center_tapped_rectifier_with_resistive_load,
    "Single-phase center-tapped transformer rectifier design with resistive load": rectifier_functionstwo.design_center_tapped_rectifier,
    "Single-phase center-tapped transformer rectifier design with RL load": rectifier_functionstwo.design_center_tapped_rectifier_with_rl_load,
    "Single-phase full-wave bridge rectifier with RL load design": rectifier_functionstwo.design_additional_series_resistance_for_electromagnet,
    "Full-wave rectifier with RL load and DC source": rectifier_functionstwo.full_wave_rectifier_with_rl_and_dc_source,
    "Single-phase full-wave bridge rectifier with RL-source load": rectifier_functionstwo.single_phase_full_wave_bridge_with_rl_source,
    "Full-wave rectifier with resistive load and capacitor filter": rectifier_functionstwo.design_fullwave_rectifier_cap_filter,
    "Full-wave rectifier with capacitive filter from Vm": rectifier_functionstwo.fullwave_rectifier_cap_filter_from_Vm,
    "Full-wave rectifier with LC filter": rectifier_functionstwo.fullwave_rectifier_lc_filter_largeC,
    "Controlled single-phase bridge rectifier with resistive load": rectifier_functionstwo.controlled_bridge_rectifier_resistive_load,
    "Controlled single-phase full-wave bridge rectifier with RL load": rectifier_functionstwo.controlled_fullwave_bridge_rl_load_avg_for_two_alphas,
    "Controlled single-phase full-wave rectifier with resistive load and isolation transformer design": rectifier_functionstwo.controlled_fullwave_rectifier_with_transformer,
    "Controlled single-phase rectifier with RL load": rectifier_functionstwo.controlled_rectifier_rl_for_required_current,
    "Full-wave converter in inverter mode with RL load": rectifier_functionstwo.fullwave_inverter_rl_power_ripple,
    "Controlled converter for solar power integration": rectifier_functionstwo.solar_converter_power_and_ripple,
    "Full-wave SCR bridge for solar power integration": rectifier_functionstwo.solar_scr_bridge_design,
    "Full-wave converter in inverter mode for wind power integration": rectifier_functionstwo.wind_inverter_design,
    "Three-phase rectifier with resistive load": rectifier_functionstwo.three_phase_rectifier_resistive,
    "Three-phase rectifier with RL load":rectifier_functionstwo.three_phase_rectifier_with_rl_load,
}


# Load JSON data
JSON_FILE = "power_electronics_questions.json"
with open(JSON_FILE, "r") as f:
    data = json.load(f)

# Few-shot examples for extraction
FEW_SHOT_EXAMPLES_EXTRACTION = "\n".join([
    f"Example {i + 1}:\nProblem: {ex['Problem']}\nClassification: {ex['Classification & Category']}\nParameters: {', '.join([f'{k}={v}' for k, v in ex['Parameters'].items()])}\nRequired Outputs: {', '.join(ex['Required Outputs'])}"
    for i, ex in enumerate(data)
])

# Prompt template for extraction
EXTRACTION_PROMPT_TEMPLATE = """
You are an AI that extracts classification, parameters, and required outputs from power electronics questions.

Output in this exact format:
Classification: <classification>
Parameters: key1=value1, key2=value2, ...
Required Outputs: output1, output2, ...

Few-shot examples:
{FEW_SHOT_EXAMPLES}

Problem: {problem}
"""

def parse_extraction_response(response):
    lines = response.split("\n")
    class_line = [l for l in lines if l.startswith("Classification:")]
    params_line = [l for l in lines if l.startswith("Parameters:")]
    outputs_line = [l for l in lines if l.startswith("Required Outputs:")]
    if not all([class_line, params_line, outputs_line]):
        raise ValueError("Invalid extraction response format")
    classification = class_line[0].split(":")[1].strip()
    params_str = params_line[0].split(":")[1].strip()
    params = {}
    for p in params_str.split(","):
        if "=" in p:
            key, value = p.split("=")
            key = key.strip()
            value = value.strip()
            try:
                # Handle list parameters like R_values or C_values
                if value.startswith("[") and value.endswith("]"):
                    value = value[1:-1].split(",")
                    params[key] = [float(v.strip()) for v in value]
                else:
                    params[key] = float(value)
            except ValueError:
                params[key] = value
    required_outputs = [o.strip() for o in outputs_line[0].split(":")[1].split(",")]
    return classification, params, required_outputs

def process_problem(problem):
    extraction_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        FEW_SHOT_EXAMPLES=FEW_SHOT_EXAMPLES_EXTRACTION,
        problem=problem
    )
    extraction_response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a precise extractor for power electronics problems."},
            {"role": "user", "content": extraction_prompt}
        ]
    )
    extraction_output = extraction_response.choices[0].message.content
    classification, params, required_outputs = parse_extraction_response(extraction_output)
    if classification in FUNCTION_MAP:
        results = FUNCTION_MAP[classification](**params)
        # Format output for Telegram
        formatted_results = []
        # Handle nested results (e.g., dual capacitor or dual resistance)
        for key, value in results.items():
            if isinstance(value, dict):
                # Check for steps_text
                if "steps_text" in value:
                    formatted_results.append(f"{key}:\n{value['steps_text']}")
                else:
                    # Collect step-related keys (e.g., "Step 1: ...", "Step 2: ...")
                    steps = [v for k, v in value.items() if k.startswith("Step")]
                    if steps:
                        formatted_results.append(f"{key}:\n" + "\n".join(steps))
                    # Include required outputs from nested dict
                    for sub_key, sub_value in value.items():
                        if sub_key in required_outputs:
                            if isinstance(sub_value, (int, float)):
                                formatted_results.append(f"{sub_key}: {sub_value:.4f}")
                            else:
                                formatted_results.append(f"{sub_key}: {sub_value}")
            elif key in required_outputs:
                # Format required outputs
                if isinstance(value, (int, float)):
                    formatted_results.append(f"{key}: {value:.4f}")
                elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    # Handle special case for Fourier series or similar lists
                    formatted_results.append(f"{key}:\n" + "\n".join(
                        [f"n={item['n']}: Vn={item['Vn']:.4f} V, Zn={item['Zn']:.4f} Ω, In={item['In']:.4f} A" for item in value]
                    ))
                else:
                    formatted_results.append(f"{key}: {value}")
            elif key.startswith("Step"):
                # Collect top-level step keys if no steps_text
                if "steps_text" not in results:
                    formatted_results.append(value)
        # If steps_text exists at top level, prepend it
        if "steps_text" in results:
            formatted_results.insert(0, results["steps_text"])
        return "\n".join(formatted_results), classification, extraction_output
    else:
        raise ValueError(f"Classification '{classification}' not found in FUNCTION_MAP")

# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to the Power Electronics Bot! Send a problem description or type /list to see available problems."
    )
def split_message(text: str, limit: int = 4000):
    """
    Split text into chunks <= limit characters.
    Prefers splitting at double newlines, then single newlines, then hard cut.
    """
    parts = []
    while len(text) > limit:
        # prefer double-newline split
        cut = text.rfind("\n\n", 0, limit)
        if cut == -1:
            cut = text.rfind("\n", 0, limit)
            if cut == -1:
                cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts

async def list_problems(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Send the full list of problems but split into Telegram-safe chunks
    to avoid BadRequest: Message is too long.
    """
    try:
        # Build the long problems string
        problem_list = "\n".join([f"{i + 1}: {p['Problem']}" for i, p in enumerate(data)])
        header = "Available problems (use /solve <number> to solve):\n\n"
        body = header + problem_list

        # Use your split_message helper to break into chunks <= limit
        for chunk in split_message(body, limit=4000):
            await update.message.reply_text(chunk)
    except Exception as e:
        # Keep errors short to avoid cascading failures
        await update.message.reply_text(f"Error while listing problems: {str(e)}")

async def solve_problem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        problem_idx = int(context.args[0]) - 1
        if 0 <= problem_idx < len(data):
            problem = data[problem_idx]["Problem"]
            formatted_results, classification, extraction_output = process_problem(problem)
            response = f"Problem: {problem}\nClassification: {classification}\nExtraction:\n{extraction_output}\nResults:\n{formatted_results}"
            await update.message.reply_text(response)
        else:
            await update.message.reply_text("Invalid problem number. Use /list to see available problems.")
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    problem = update.message.text
    try:
        formatted_results, classification, extraction_output = process_problem(problem)
        response = f"Problem: {problem}\nClassification: {classification}\nExtraction:\n{extraction_output}\nResults:\n{formatted_results}"
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("list", list_problems))
    application.add_handler(CommandHandler("solve", solve_problem))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()

if __name__ == "__main__":
    main()