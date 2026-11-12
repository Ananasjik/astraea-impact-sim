class Input {
    constructor(canvas) {
        this.keys = {};
        this.keysJustPressed = {};
        
        this.mouseX = 0;
        this.mouseY = 0;
        this.mouseButtons = {};
        this.mouseJustPressed = {};
        
        // Prevent context menu
        canvas.addEventListener('contextmenu', e => e.preventDefault());

        window.addEventListener('keydown', (e) => {
            if (!this.keys[e.code]) {
                this.keysJustPressed[e.code] = true;
            }
            this.keys[e.code] = true;
        });

        window.addEventListener('keyup', (e) => {
            this.keys[e.code] = false;
        });

        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            this.mouseX = e.clientX - rect.left;
            this.mouseY = e.clientY - rect.top;
        });

        canvas.addEventListener('mousedown', (e) => {
            if (!this.mouseButtons[e.button]) {
                this.mouseJustPressed[e.button] = true;
            }
            this.mouseButtons[e.button] = true;
        });

        canvas.addEventListener('mouseup', (e) => {
            this.mouseButtons[e.button] = false;
        });
    }

    isKeyDown(code) {
        return !!this.keys[code];
    }

    isKeyJustPressed(code) {
        return !!this.keysJustPressed[code];
    }

    isMouseDown(button) {
        return !!this.mouseButtons[button];
    }

    isMouseJustPressed(button) {
        return !!this.mouseJustPressed[button];
    }

    isMouseOverUI() {
        // Very simple check: if mouse is over UI elements (which have pointer-events: auto)
        // This is a naive implementation, actual hit testing might be needed for a real game.
        // For now we'll check if the event target was a UI element before the canvas catches it,
        // but since we attach events to window/canvas, we can just assume if the user is hovering a DOM element not named Canvas.
        const el = document.elementFromPoint(this.mouseX, this.mouseY);
        return el && el.tagName !== 'CANVAS';
    }

    postUpdate() {
        this.keysJustPressed = {};
        this.mouseJustPressed = {};
    }
}
