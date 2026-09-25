import os
import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from PIL import Image

def generate_clinical_pdf(
    patient_id: str,
    prediction_label: str,
    confidence: float,
    morphology_data: dict,
    original_pil: Image.Image,
    cam_pil: Image.Image,
    seg_pil: Image.Image,
    output_path: str = None
) -> bytes:
    """
    Generates a PDF Clinical Decision Support Report for Leukemia Smear Analysis.
    Returns bytes and optionally writes to output_path.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        output_path if output_path else buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography & Colors
    primary_color = colors.HexColor("#0f172a") # Deep slate
    accent_color = colors.HexColor("#0284c7")  # Medical cyan
    danger_color = colors.HexColor("#dc2626")  # Crimson
    success_color = colors.HexColor("#16a34a") # Emerald
    light_bg = colors.HexColor("#f8fafc")

    nc_val = float(morphology_data.get('nc_ratio', 0.0))
    is_elevated_nc = (nc_val >= 0.70)
    is_all = "Leukemia" in prediction_label or "ALL" in prediction_label or is_elevated_nc

    warning_color = colors.HexColor("#d97706") # Amber
    if is_all and not is_elevated_nc:
        status_color = danger_color
        banner_color = colors.HexColor("#fef2f2")
        border_color = danger_color
        risk_title = "CRITICAL FINDING: ACUTE LYMPHOBLASTIC LEUKEMIA (ALL) DETECTED"
    elif is_elevated_nc and "Leukemia" not in prediction_label and "ALL" not in prediction_label:
        status_color = warning_color
        banner_color = colors.HexColor("#fffbeb")
        border_color = warning_color
        risk_title = "ATYPICAL MORPHOLOGY / HIGH N:C RATIO — PATHOLOGIST REVIEW REQUIRED"
    elif is_all:
        status_color = danger_color
        banner_color = colors.HexColor("#fef2f2")
        border_color = danger_color
        risk_title = "CRITICAL FINDING: ACUTE LYMPHOBLASTIC LEUKEMIA (ALL) DETECTED"
    else:
        status_color = success_color
        banner_color = colors.HexColor("#f0fdf4")
        border_color = success_color
        risk_title = "CYTOLOGICALLY NORMAL / BENIGN LYMPHOCYTE"

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        alignment=TA_LEFT
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_LEFT
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceAfter=6
    )

    cell_body = ParagraphStyle(
        'CellBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155")
    )

    cell_body_bold = ParagraphStyle(
        'CellBodyBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0f172a")
    )

    elements = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>HemaVision AI Diagnostics</b><br/><font size=8 color='#64748b'>Automated Cytological Intelligence & Peripheral Blood Smear Decision Support</font>", title_style),
            Paragraph(f"<b>REPORT ID:</b> HV-{datetime.datetime.now().strftime('%Y%m%d%H%M')}<br/><b>DATE:</b> {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[380, 160])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (1,0), (1,0), 'RIGHT')
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceAfter=12))

    # 2. Patient & Specimen Info Table
    info_data = [
        [
            Paragraph("<b>Patient Identifier:</b>", cell_body_bold),
            Paragraph(patient_id or "ANON-SPECIMEN-01", cell_body),
            Paragraph("<b>Specimen Type:</b>", cell_body_bold),
            Paragraph("Peripheral Blood Smear (WBC)", cell_body),
        ],
        [
            Paragraph("<b>Staining Protocol:</b>", cell_body_bold),
            Paragraph("Wright-Giemsa Cytochemical Stain", cell_body),
            Paragraph("<b>Magnification:</b>", cell_body_bold),
            Paragraph("100x Oil Immersion Optical Microscopy", cell_body),
        ],
        [
            Paragraph("<b>Algorithm Engine:</b>", cell_body_bold),
            Paragraph("Deep CNN + Grad-CAM Saliency", cell_body),
            Paragraph("<b>Clinical Validation:</b>", cell_body_bold),
            Paragraph("Automated Pre-Screening Flag", cell_body),
        ]
    ]
    info_table = Table(info_data, colWidths=[120, 150, 120, 150])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#f1f5f9")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 14))

    # 3. Primary Diagnostic Result Banner
    diag_summary = [
        [
            Paragraph(f"<font color='{status_color.hexval()}'><b>{risk_title}</b></font><br/>"
                      f"<font size=9 color='#475569'>Model Classification Confidence: <b>{confidence * 100:.1f}%</b> | "
                      f"Morphology Assessment: <b>{morphology_data.get('morphology_status', 'Evaluated')}</b></font>", cell_body)
        ]
    ]
    diag_table = Table(diag_summary, colWidths=[540])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), banner_color),
        ('BOX', (0,0), (-1,-1), 1.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    elements.append(diag_table)
    elements.append(Spacer(1, 14))

    # 4. Multi-Modal Visual Cytometry Panel (Original Smear, Grad-CAM, Segmentation)
    elements.append(Paragraph("<b>1. Multi-Modal Microscopic Cytometry Analysis</b>", section_heading))
    
    # Save temporary images to in-memory buffers
    def pil_to_rl(im, size=(165, 140)):
        b = io.BytesIO()
        im.resize((300, 260)).save(b, format='JPEG', quality=95)
        b.seek(0)
        return RLImage(b, width=size[0], height=size[1])

    img1 = pil_to_rl(original_pil)
    img2 = pil_to_rl(cam_pil)
    img3 = pil_to_rl(seg_pil)

    image_grid_data = [
        [img1, img2, img3],
        [
            Paragraph("<b>(A) Original Smear</b><br/><font size=7 color='#64748b'>Wright-Giemsa 100x View</font>", cell_body),
            Paragraph("<b>(B) Grad-CAM Saliency</b><br/><font size=7 color='#64748b'>Nuclear Chromatin Heatmap</font>", cell_body),
            Paragraph("<b>(C) Cytoplasm/Nucleus Mask</b><br/><font size=7 color='#64748b'>Morphological Segmentation</font>", cell_body)
        ]
    ]
    image_table = Table(image_grid_data, colWidths=[180, 180, 180])
    image_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(image_table)
    elements.append(Spacer(1, 12))

    # 5. Quantitative Cytometry Metrics Table
    elements.append(Paragraph("<b>2. Computational Morphometric Biomarkers</b>", section_heading))
    
    nc_val = morphology_data.get('nc_ratio', 0.0)
    nc_status = "ABNORMAL / ELEVATED (Blast Cell Indicator)" if nc_val >= 0.70 else "NORMAL PHYSIOLOGICAL RANGE"
    
    metrics_data = [
        [Paragraph("<b>Cytological Parameter</b>", cell_body_bold), 
         Paragraph("<b>Measured Value</b>", cell_body_bold), 
         Paragraph("<b>Reference Normal Range</b>", cell_body_bold),
         Paragraph("<b>Clinical Correlation</b>", cell_body_bold)],
        
        [Paragraph("Nucleus-to-Cytoplasmic (N:C) Ratio", cell_body),
         Paragraph(f"<b>{nc_val:.3f}</b>", cell_body),
         Paragraph("0.40 - 0.60", cell_body),
         Paragraph(nc_status, cell_body)],
        
        [Paragraph("Nuclear Circularity / Regularity", cell_body),
         Paragraph(f"{morphology_data.get('nuclear_circularity', 0.0):.3f}", cell_body),
         Paragraph("0.85 - 1.00", cell_body),
         Paragraph("Round/Uniform vs. Cleaved/Pleomorphic", cell_body)],
         
        [Paragraph("Nucleus Pixel Footprint", cell_body),
         Paragraph(f"{morphology_data.get('nucleus_area_pixels', 0):,} px", cell_body),
         Paragraph("Standard 45-60 µm²", cell_body),
         Paragraph("Hyperchromatic Blast Nucleus" if (is_all or is_elevated_nc) else "Normal Mature Nucleus", cell_body)],

        [Paragraph("Cellular Risk Stratification", cell_body),
         Paragraph(f"<b>{'HIGH (Category 3)' if is_all and ('Leukemia' in prediction_label or 'ALL' in prediction_label) else ('SUSPICIOUS (Category 2)' if is_elevated_nc else 'LOW (Category 1)')}</b>", cell_body),
         Paragraph("Category 1 (Negative)", cell_body),
         Paragraph("Urgent Bone Marrow Biopsy Suggested" if (is_all and ('Leukemia' in prediction_label or 'ALL' in prediction_label)) else ("Pathologist Review & Flow Cytometry Mandated" if is_elevated_nc else "Routine Annual CBC"), cell_body)]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[150, 110, 130, 150])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 14))

    # 6. Pathologist Recommendations & Legal Disclaimer
    elements.append(Paragraph("<b>3. Interpretive Clinical Observations & Next Steps</b>", section_heading))
    if is_all and ("Leukemia" in prediction_label or "ALL" in prediction_label):
        comments_text = (
            "<b>Recommendations:</b> High suspicion of Acute Lymphoblastic Leukemia lymphoblasts based on deep neural network activation and elevated N:C ratio. "
            "Recommend immediate confirmatory Flow Cytometry (CD10, CD19, TdT immunophenotyping) and Hematopathologist bone marrow aspiration."
        )
    elif is_elevated_nc:
        comments_text = (
            f"<b>Recommendations:</b> Specimen exhibits marked nuclear expansion with an elevated N:C ratio of <b>{nc_val:.3f}</b> (normal reference: 0.40 - 0.60). "
            "Because high N:C ratio is a definitive cytopathological blast hallmark, this specimen cannot be declared benign normal. "
            "Urgent manual hematopathologist microscopic correlation and Complete Blood Count (CBC) with differential are strongly indicated."
        )
    else:
        comments_text = (
            "<b>Recommendations:</b> Smear exhibits mature lymphocytic profile without suspicious lymphoblast morphology. "
            "Correlate with Complete Blood Count (CBC) and absolute lymphocyte counts."
        )
    elements.append(Paragraph(comments_text, cell_body))
    elements.append(Spacer(1, 10))

    disclaimer = (
        "<font size=7 color='#94a3b8'><b>RESEARCH AND DECISION SUPPORT NOTICE:</b> This report was generated automatically by the HemaVision AI Deep Learning Pipeline. "
        "Intended for research, screening, and educational decision-support purposes. Final medical diagnosis must always be confirmed by a certified board hematopathologist.</font>"
    )
    elements.append(Paragraph(disclaimer, cell_body))
    
    # Build Document
    doc.build(elements)
    
    if output_path:
        with open(output_path, "rb") as f:
            return f.read()
            
    return buffer.getvalue()
