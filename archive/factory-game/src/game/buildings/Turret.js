class Turret extends Building {
    constructor(x, y) {
        super(x, y, 'turret');
        this.range = 150;
        this.cooldown = 1000;
        this.timer = 0;
        this.target = null;
        // Turrets normally need ammo, but we make it shoot free magic bullets for proto
    }

    update(dt, world) {
        this.timer -= dt;
        
        // Find target
        const enemies = window.game.enemies.list;
        this.target = null;
        let minDist = this.range;
        
        const px = this.x * TILE_SIZE + TILE_SIZE/2;
        const py = this.y * TILE_SIZE + TILE_SIZE/2;
        
        for (const e of enemies) {
            const dx = e.x - px;
            const dy = e.y - py;
            const dist = Math.sqrt(dx*dx + dy*dy);
            if (dist < minDist) {
                minDist = dist;
                this.target = e;
            }
        }

        if (this.target && this.timer <= 0) {
            // Shoot
            this.target.health -= 10;
            this.timer = this.cooldown;
            
            // Draw a quick line (not best practice in update, but good for proto)
            const ctx = window.game.ctx;
            ctx.save();
            window.game.camera.apply(ctx);
            ctx.strokeStyle = '#ffff00';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(px, py);
            ctx.lineTo(this.target.x, this.target.y);
            ctx.stroke();
            ctx.restore();
        }
    }
}
