from flask import Flask, render_template, request
import pandas as pd
import os
from datetime import datetime
from flask import send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from analytics import (
    calculate_metrics,
    clean_data,
    data_quality_after,
    data_quality_before,
    detect_outliers,
    get_kpis,
    preprocess_data,
    revenue_chart,
    roi_chart,
    pie_chart,
    generate_insights,
    data_quality_report,
    train_roi_model,
    feature_importance_chart,
    generate_recommendations,
)

app = Flask(__name__)

latest_kpis = {}

latest_before_report = {}
latest_after_report = {}

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def home():

    global latest_kpis
    if request.method == "POST":

        file = request.files["file"]

        if file:

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            analysis_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

            file.save(filepath)

            df = pd.read_csv(filepath)

            before_report = data_quality_before(df)

            df = preprocess_data(df)

            after_report = data_quality_after(df)

            global latest_before_report
            global latest_after_report

            latest_before_report = before_report
            latest_after_report = after_report

            outlier_count = detect_outliers(df)

            df = clean_data(df)

            df = calculate_metrics(df)

            predicted_roi, accuracy, best_model, model_scores = train_roi_model(df)

            importance_chart = feature_importance_chart(df)

            kpis = get_kpis(df)

            latest_kpis = kpis

            chart = revenue_chart(df)

            roi = roi_chart(df)

            insights = generate_insights(df)

            recommendations = generate_recommendations(df)

            pie = pie_chart(df)

            table = df.head(20).to_html(
    index=False,
    classes="custom-table",
    border=0
)
            
            filename = file.filename

            return render_template(
                "dashboard.html",
                kpis=kpis,
                chart=chart,
                roi_chart=roi,
                insights=insights,
                pie_chart=pie,
                table=table,
                dataset_name=filename,
                before_report=before_report,
                after_report=after_report,
                analysis_time=analysis_time,
                outlier_count=outlier_count,
                predicted_roi=predicted_roi,
                accuracy=accuracy,
                importance_chart=importance_chart,
                recommendations=recommendations,
                best_model=best_model,
                model_scores=model_scores
                )

    return render_template("upload.html")

@app.route("/download-report")
def download_report():

    global latest_kpis
    global latest_before_report
    global latest_after_report

    pdf = SimpleDocTemplate("report.pdf")

    pdf.title = "Marketing ROI Analysis Report"

    pdf.author = "Daksh Jain"

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
    "TitleStyle",
    parent=styles["Title"],
    alignment=TA_CENTER,
    textColor=colors.HexColor("#2563eb"),
    fontSize=26,
    leading=30,
    spaceAfter=20
)
    section_style = ParagraphStyle(
    "SectionStyle",
    parent=styles["Heading2"],
    alignment=TA_CENTER,
    textColor=colors.black,
)

    text_style = ParagraphStyle(
    "TextStyle",
    parent=styles["Normal"],
    textColor=colors.HexColor("#334155")
)

    title_style = ParagraphStyle(
    "TitleStyle",
    parent=styles["Title"],
    alignment=TA_CENTER,
    textColor=colors.HexColor("#2563eb")
)

    heading_style = ParagraphStyle(
    "HeadingStyle",
    parent=styles["Heading2"],
    textColor=colors.black,
)

    normal_style = ParagraphStyle(
    "NormalStyle",
    parent=styles["Normal"],
    textColor=colors.HexColor("#334155")
)

    content = []

    content.append(
    Paragraph(
        "Marketing ROI Analysis Report",
        title_style
    )
)

    content.append(Spacer(1,20))

    content.append(
    Paragraph(
        "<b>Executive Summary</b>",
        heading_style
    )
)
    
    content.append(
    Paragraph(
        "<b>Report Information</b>",
        heading_style
    )
)

    content.append(
    Paragraph(
        f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        normal_style
    )
)

    content.append(
    Paragraph(
        "Generated By: Marketing ROI Analyzer",
        normal_style
    )
)

    content.append(Spacer(1,15))

    content.append(
    Paragraph(
        "This report summarizes marketing campaign performance, data quality analysis, machine learning predictions and business insights.",
        normal_style
    )
)

    content.append(Spacer(1,15))

    content.append(
    Paragraph(
        "<b>Key Performance Indicators</b>",
        heading_style
    )
)

    content.append(
    Paragraph(
        f"Total Revenue : Rs. {latest_kpis.get('total_revenue',0)}",
        normal_style
    )
)

    content.append(
    Paragraph(
        f"Total Spend : Rs. {latest_kpis.get('total_spend',0)}",
        normal_style
    )
)

    content.append(
    Paragraph(
        f"Total Profit : Rs. {latest_kpis.get('total_profit',0)}",
        normal_style
    )
)

    content.append(
    Paragraph(
        f"Average ROI : {latest_kpis.get('avg_roi',0)}%",
        styles["Normal"]
    )
)
    
    content.append(Spacer(1,15))

    content.append(
    Paragraph(
        "Business Metrics",
        section_style
    )
)

    kpi_data = [
    ["Metric", "Value"],
    ["Total Revenue", f"Rs. {latest_kpis.get('total_revenue',0)}"],
    ["Total Spend", f"Rs. {latest_kpis.get('total_spend',0)}"],
    ["Total Profit", f"Rs. {latest_kpis.get('total_profit',0)}"],
    ["Average ROI", f"{latest_kpis.get('avg_roi',0)}%"]
]

    kpi_table = Table(kpi_data)

    kpi_table.setStyle(
    TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1e293b")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),

        ("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#f8fafc")),

        ("GRID",(0,0),(-1,-1),1,colors.HexColor("#cbd5e1")),

        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

        ("BOTTOMPADDING",(0,0),(-1,0),12),

        ("ALIGN",(0,0),(-1,-1),"CENTER")
    ])
)

    content.append(kpi_table)

    content.append(Spacer(1,20))

    content.append(
    Paragraph(
        "Data Quality Report",
        section_style
    )
)
    quality_data = [
    ["Stage","Missing","Duplicates","Outliers"],
    [
        "Before Cleaning",
        latest_before_report.get("missing",0),
        latest_before_report.get("duplicates",0),
        latest_before_report.get("outliers",0)
    ],
    [
        "After Cleaning",
        latest_after_report.get("missing",0),
        latest_after_report.get("duplicates",0),
        latest_after_report.get("outliers",0)
    ]
]

    quality_table = Table(quality_data)

    quality_table.setStyle(
    TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1e293b")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#f8fafc")),
        ("GRID",(0,0),(-1,-1),1,colors.HexColor("#cbd5e1")),
        ("ALIGN",(0,0),(-1,-1),"CENTER")
    ])
)

    content.append(quality_table)

    content.append(Spacer(1,20))

#     content.append(
#     Paragraph(
#         f"Before Cleaning → Missing Values: {latest_before_report.get('missing',0)} | Duplicates: {latest_before_report.get('duplicates',0)} | Outliers: {latest_before_report.get('outliers',0)}",
#         normal_style
#     )
# )

#     content.append(
#     Paragraph(
#         f"After Cleaning → Missing Values: {latest_after_report.get('missing',0)} | Duplicates: {latest_after_report.get('duplicates',0)} | Outliers: {latest_after_report.get('outliers',0)}",
#         normal_style
#     )
# )

#     content.append(Spacer(1,15))

    content.append(Spacer(1,15))

    content.append(
    Paragraph(
        "<b>Project Capabilities</b>",
        heading_style
    )
)

    content.append(
    Paragraph(
        "• Data Quality Detection",
        normal_style
    )
)

    content.append(
    Paragraph(
        "• Missing Value Handling",
        normal_style
    )
)

    content.append(
    Paragraph(
        "• Duplicate Record Removal",
        normal_style
    )
)

    content.append(
    Paragraph(
        "• Outlier Detection using IQR",
        normal_style
    )
)

    content.append(
    Paragraph(
        "• ROI Prediction using Random Forest",
        normal_style
    )
)

    content.append(
    Paragraph(
        "• AI Generated Business Recommendations",
        normal_style
    )
)

    pdf.build(content)

    return send_file(
        "report.pdf",
        as_attachment=True,
        download_name="Marketing_ROI_Analysis_Report.pdf"
    )

if __name__ == "__main__":
    app.run(debug=True)

   