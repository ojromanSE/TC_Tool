# Add plots
        hist_png, box_png, scatter_png, _ = bfactor_analytics_figures(oneline, fluid_name, eur_col)
        
        if hist_png:
            story.append(Paragraph("B-Factor Distribution", styles['Heading2']))
            story.append(Spacer(1, 6))
            story.append(Image(hist_png, width=5*inch, height=2.5*inch))
            story.append(Spacer(1, 12))
        
        # Create a new page for the three plots together
        story.append(PageBreak())
        story.append(Paragraph(f"{fluid_name} Analysis Charts", heading_style))
        story.append(Spacer(1, 12))
        
        # EUR vs B-Factor
        if scatter_png:
            story.append(Paragraph("EUR vs B-Factor", styles['Heading2']))
            story.append(Spacer(1, 6))
            story.append(Image(scatter_png, width=5.5*inch, height=2.6*inch))
            story.append(Spacer(1, 10))
        
        # Type curves
        if monthly_key in st.session_state:
            curves, lines = build_type_curves_and_lines(st.session_state[monthly_key], fluid_name.lower())
            if not curves.empty:
                story.append(Paragraph("Type Curve", styles['Heading2']))
                story.append(Spacer(1, 6))
                fig = plot_type_curves(curves, lines, fluid_name.lower())
                tc_png = _save_fig(fig)
                story.append(Image(tc_png, width=5.5*inch, height=2.6*inch))
                story.append(Spacer(1, 10))
        
        # Probit plot
        if eur_col in oneline.columns:
            eurs = pd.to_numeric(oneline[eur_col], errors="coerce").dropna().astype(float).tolist()
            if fluid_name == "Gas":
                eurs = [x * 1000 for x in eurs]
            if eurs:
                unit = "Mbbl" if fluid_name != "Gas" else "MMcf"
                story.append(Paragraph("Probit Plot", styles['Heading2']))
                story.append(Spacer(1, 6))
                fig = probit_plot(eurs, unit, f"{fluid_name} EUR Distribution", _phase_color(fluid_name))
                probit_png = _save_fig(fig)
                story.append(Image(probit_png, width=5.5*inch, height=2.6*inch))
                story.append(Spacer(1, 12))
        
        story.append(PageBreak())
