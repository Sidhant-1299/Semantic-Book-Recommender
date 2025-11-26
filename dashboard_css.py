custom_css = """
body { background-color: #111; color: #fff; font-family: 'Helvetica', sans-serif; }

/* Header */
.header { text-align: center; margin-bottom: 40px; }
.header h1 { font-size: 3rem; margin: 0; }
.header p { color: #ccc; font-size: 1.2rem; }

/* Inputs */
.inputs { display: flex; gap: 15px; justify-content: center; margin-bottom: 40px; }
.gr-textbox, .gr-dropdown { border-radius: 6px !important; padding: 10px; background-color: #222 !important; color: #fff; }

/* Button */
.custom-btn {
    background: #e50914 !important;
    border-radius: 6px;
    padding: 12px 24px;
    font-weight: bold;
    transition: transform 0.2s, box-shadow 0.2s;
}
.custom-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 15px rgba(0,0,0,0.6);
}

/* Gallery as horizontal scroll row */
.gr-gallery { display: flex; overflow-x: auto; gap: 20px; padding-bottom: 20px; scroll-behavior: smooth; }
.gr-gallery img { border-radius: 8px; height: 300px; transition: transform 0.2s, filter 0.2s; cursor: pointer; }
.gr-gallery img:hover { transform: scale(1.1); filter: brightness(1.2); }

.modal {
    position: fixed;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    background: #222;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.8);
    max-width: 600px;
    z-index: 9999;

.modal h2 { margin-top: 0; color: #fff; }
.modal p { color: #ccc; line-height: 1.5; }


"""

