class Camera {
    constructor(width, height) {
        this.x = 0;
        this.y = 0;
        this.width = width;
        this.height = height;
        this.zoom = 1; // Not implemented zooming yet, but good to have
    }

    resize(width, height) {
        this.width = width;
        this.height = height;
    }

    follow(targetX, targetY) {
        // Smooth follow
        const destX = targetX - this.width / 2;
        const destY = targetY - this.height / 2;
        
        this.x += (destX - this.x) * 0.1;
        this.y += (destY - this.y) * 0.1;
    }

    screenToWorld(screenX, screenY) {
        return {
            x: (screenX / this.zoom) + this.x,
            y: (screenY / this.zoom) + this.y
        };
    }

    worldToScreen(worldX, worldY) {
        return {
            x: (worldX - this.x) * this.zoom,
            y: (worldY - this.y) * this.zoom
        };
    }

    apply(ctx) {
        ctx.scale(this.zoom, this.zoom);
        ctx.translate(-this.x, -this.y);
    }
}
