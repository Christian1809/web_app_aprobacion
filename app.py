import gradio as gr
from bank_ml import BankModel

bank_model = BankModel()

def predict_ui(job, age, campaign, euribor3m, threshold):
    result = bank_model.predict_loan(
        job=job,
        age=age,
        campaign=campaign,
        euribor3m=euribor3m,
        threshold=threshold
    )

    decision = "✅ ACEPTA el préstamo" if result["prediction"] == 1 else "❌ NO acepta el préstamo"

    return f"""{decision}
Probabilidad de aceptación: {result['probability']:.2%}
"""

iface = gr.Interface(
    fn=predict_ui,
    inputs=[
        gr.Dropdown(
            ['admin.', 'blue-collar', 'technician', 'management', 'services',
             'retired', 'self-employed', 'unemployed', 'housemaid', 'student'],
            label="Ocupación"
        ),
        gr.Slider(18, 80, value=35, label="Edad"),
        gr.Slider(1, 10, step=1, value=1, label="Campaign"),
        gr.Slider(0.0, 6.0, value=1.0, label="Euribor 3 meses"),
        gr.Slider(0.1, 0.6, value=0.25, label="Umbral")
    ],
    outputs=gr.Textbox(label="Resultado"),
    title="Simulador de Aceptación de Préstamo"
)

iface.launch()