custom_css = """
/* --- GLOBAL THEME --- */
body, .gradio-container { 
    background-color: #141414 !important; /* Netflix Black */
    color: #e5e5e5 !important; 
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; 
}

/* --- HEADER --- */
.header { 
    text-align: center; 
    padding: 60px 20px 20px 20px; 
}
.header h1 { 
    font-size: 3.5rem; 
    font-weight: 700; 
    color: #e50914; /* Netflix Red */
    margin: 0;
}
.header p { 
    color: #a3a3a3; 
    font-size: 1.2rem; 
    margin-top: 10px; 
}

/* --- WIDE & INVISIBLE INPUT CONTAINER --- */
.inputs { 
    max-width: 1200px !important; /* Make it Wide */
    width: 100% !important;
    margin: 0 auto 50px auto !important; 
    background: transparent !important; /* Into the background */
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    gap: 15px;
}

/* Style the actual input boxes to pop against the background */
.gr-textbox, .gr-dropdown {
    background-color: rgba(50, 50, 50, 0.8) !important;
    border: 1px solid #444 !important;
    border-radius: 4px !important;
}
/* Text inside inputs */
.gr-textbox input, .gr-dropdown span {
    color: white !important;
    font-size: 1.1rem;
}

.custom-btn {
    background: #e50914 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: bold;
    font-size: 1rem;
    height: 100%; 
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease;
}
.custom-btn:hover {
    background: #f40612 !important;
    box-shadow: 0 0 15px rgba(229, 9, 20, 0.6);
}

/* --- GALLERY --- */
.gr-gallery {
    padding: 20px;
}
.gallery-item img {
    border-radius: 4px;
    transition: transform 0.3s ease;
}
.gallery-item:hover img {
    transform: scale(1.05);
    z-index: 10;
}

/* --- MODAL FIXES --- */

/* 1. The Backdrop */
.modal {
    background: rgba(0, 0, 0, 0.85) !important;
    backdrop-filter: blur(8px) !important;
    padding: 0 !important; /* Removes the gap issue */
    display: flex !important;
    align-items: center;
    justify-content: center;
}

/* 2. Hide the DEFAULT close button (The small 'x') */
/* We target any button that is NOT our custom close-btn */
.modal > button:not(#close-btn) {
    display: none !important;
}

/* 3. The Card Container (The visible part) */
.modal-body { 
    background-color: #181818 !important;
    border-radius: 10px !important;
    box-shadow: 0 20px 50px rgba(0,0,0,0.9);
    border: 1px solid #333;
    
    /* Size Controls */
    width: 500px !important;
    max-width: 90vw !important;
    max-height: 90vh !important;
    
    /* Layout */
    display: flex;
    flex-direction: column;
    overflow: hidden; /* Ensures image corners stay rounded */
    padding: 0 !important;
    margin: 0 !important;
    position: relative;
}

/* 4. Our Custom Floating Close Button */
#close-btn {
    position: absolute !important;
    top: 15px;
    right: 15px;
    z-index: 10000; /* Top of everything */
    background: rgba(0,0,0,0.6) !important;
    color: white !important;
    border-radius: 50% !important;
    width: 36px !important;
    height: 36px !important;
    border: 2px solid rgba(255,255,255,0.5) !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
    box-shadow: none !important;
}
#close-btn:hover {
    background: white !important;
    color: black !important;
    border-color: white !important;
}
.modal {
    display: flex !important;
    align-items: center;       /* Centers vertically */
    justify-content: center;   /* Centers horizontally */
    background: rgba(0, 0, 0, 0.8) !important; /* Darken background */
    backdrop-filter: blur(5px) !important;     /* Blur background */
    padding: 0 !important;     /* CRITICAL: Removes the gray gap */
}

/* 2. Hide the DUPLICATE Close Button */
/* This finds the default button created by the library and hides it */
.modal > button:not(#close-btn) {
    display: none !important;
}

/* 3. The Card (The actual popup box) */
.modal-body {
    position: relative;        /* Allows us to float the close button inside */
    background-color: #181818 !important; /* Netflix card gray */
    width: 500px !important;   /* Fixed beautiful width */
    max-width: 90vw !important; /* Responsive on mobile */
    max-height: 90vh !important;
    border-radius: 12px !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.9) !important;
    border: 1px solid #333;
    overflow: hidden;          /* Keeps image corners rounded */
    padding: 0 !important;     /* Image touches the edges */
    display: flex;
    flex-direction: column;
}

/* 4. The Custom Close Button (The 'X') */
#close-btn {
    position: absolute !important; /* Float it */
    top: 15px;
    right: 15px;
    z-index: 100 !important;      /* Sit on top of the image */
    
    /* Styling */
    background: rgba(0, 0, 0, 0.6) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 50% !important; /* Make it a circle */
    width: 32px !important;
    height: 32px !important;
    min-width: unset !important;
    font-size: 14px !important;
    padding: 0 !important;
    display: flex;
    align-items: center;
    justify-content: center;
}

#close-btn:hover {
    background: white !important;
    color: black !important;
}

/* 5. Content Styling */
/* Force the image to cover the top area */
.modal-body img {
    width: 100%;
    height: 300px; /* Fixed height for consistency */
    object-fit: cover;
    display: block;
}

/* Padding for the text below the image */
.modal-body h2 {
    padding: 20px 20px 5px 20px;
    margin: 0;
    font-size: 1.5rem;
    color: white;
}
.modal-body p {
    padding: 0 20px 20px 20px;
    color: #ccc;
    font-size: 0.95rem;
    line-height: 1.5;
    max-height: 200px; /* Scroll if description is huge */
    overflow-y: auto;
}
"""