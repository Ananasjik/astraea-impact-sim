class Renderer {
    constructor(ctx, camera) {
        this.ctx = ctx;
        this.camera = camera;
    }

    drawWorld(world) {
        const startCol = Math.floor(this.camera.x / TILE_SIZE);
        const endCol = startCol + Math.ceil(this.camera.width / TILE_SIZE) + 1;
        const startRow = Math.floor(this.camera.y / TILE_SIZE);
        const endRow = startRow + Math.ceil(this.camera.height / TILE_SIZE) + 1;

        for (let y = startRow; y <= endRow; y++) {
            for (let x = startCol; x <= endCol; x++) {
                const tile = world.getTile(x, y);
                this.drawTile(x, y, tile);
            }
        }
    }

    drawTile(x, y, tile) {
        const px = x * TILE_SIZE;
        const py = y * TILE_SIZE;

        // Base terrain
        if (tile.type === 'grass') {
            this.ctx.fillStyle = '#2d5a27';
        } else if (tile.type === 'stone') {
            this.ctx.fillStyle = '#555';
        } else if (tile.type === 'water') {
            this.ctx.fillStyle = '#2b5b84';
        } else {
            this.ctx.fillStyle = '#111';
        }
        this.ctx.fillRect(px, py, TILE_SIZE, TILE_SIZE);
        
        // Grid lines (optional, for debug)
        this.ctx.strokeStyle = 'rgba(0,0,0,0.1)';
        this.ctx.strokeRect(px, py, TILE_SIZE, TILE_SIZE);

        // Ores
        if (tile.ore) {
            this.ctx.fillStyle = this.getOreColor(tile.ore);
            // Draw a smaller rectangle for ore
            this.ctx.fillRect(px + 4, py + 4, TILE_SIZE - 8, TILE_SIZE - 8);
        }
    }

    getOreColor(oreType) {
        switch(oreType) {
            case 'coal': return '#222';
            case 'iron': return '#8a9b9e';
            case 'copper': return '#c4693b';
            default: return '#f0f';
        }
    }

    drawBuildings(buildings) {
        for (const b of buildings) {
            const px = b.x * TILE_SIZE;
            const py = b.y * TILE_SIZE;
            
            // Draw base
            this.ctx.fillStyle = '#777';
            this.ctx.fillRect(px + 2, py + 2, TILE_SIZE - 4, TILE_SIZE - 4);
            
            // Draw specific building details
            if (b.type === 'miner') {
                this.ctx.fillStyle = '#aa3333';
                this.ctx.beginPath();
                this.ctx.arc(px + TILE_SIZE/2, py + TILE_SIZE/2, TILE_SIZE/3, 0, Math.PI * 2);
                this.ctx.fill();
            } else if (b.type === 'conveyor') {
                // Direction arrow
                this.ctx.fillStyle = '#aaa';
                this.ctx.save();
                this.ctx.translate(px + TILE_SIZE/2, py + TILE_SIZE/2);
                this.ctx.rotate(b.direction * Math.PI / 2);
                this.ctx.beginPath();
                this.ctx.moveTo(-4, -6);
                this.ctx.lineTo(4, 0);
                this.ctx.lineTo(-4, 6);
                this.ctx.fill();
                this.ctx.restore();
                
                // Draw items on belt
                for (const item of b.items) {
                    let ix = px;
                    let iy = py;
                    const offset = item.position * TILE_SIZE;
                    if (b.direction === 0) ix += offset;
                    else if (b.direction === 1) iy += offset;
                    else if (b.direction === 2) ix += (TILE_SIZE - offset);
                    else if (b.direction === 3) iy += (TILE_SIZE - offset);
                    
                    const itemDef = ITEMS[item.itemId];
                    this.ctx.fillStyle = itemDef ? itemDef.icon : '#f0f';
                    this.ctx.fillRect(ix + 6, iy + 6, TILE_SIZE - 12, TILE_SIZE - 12);
                }
            } else if (b.type === 'smelter') {
                this.ctx.fillStyle = '#ff5500';
                this.ctx.fillRect(px + 4, py + 4, TILE_SIZE - 8, TILE_SIZE - 8);
            } else if (b.type === 'turret') {
                this.ctx.fillStyle = '#4444ff';
                this.ctx.beginPath();
                this.ctx.arc(px + TILE_SIZE/2, py + TILE_SIZE/2, TILE_SIZE/2.5, 0, Math.PI * 2);
                this.ctx.fill();
            }
        }
    }

    drawEnemies(enemies) {
        if (!enemies || !enemies.list) return;
        this.ctx.fillStyle = '#ff0000';
        for (const e of enemies.list) {
            this.ctx.beginPath();
            this.ctx.arc(e.x, e.y, e.radius, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    drawPlayer(player) {
        this.ctx.fillStyle = '#3388ff';
        this.ctx.beginPath();
        this.ctx.arc(player.x, player.y, player.radius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Draw facing direction
        this.ctx.strokeStyle = '#fff';
        this.ctx.beginPath();
        this.ctx.moveTo(player.x, player.y);
        this.ctx.lineTo(player.x + Math.cos(player.angle) * player.radius * 1.5, 
                        player.y + Math.sin(player.angle) * player.radius * 1.5);
        this.ctx.stroke();
    }

    drawHighlight(tileX, tileY) {
        const px = tileX * TILE_SIZE;
        const py = tileY * TILE_SIZE;
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(px, py, TILE_SIZE, TILE_SIZE);
        this.ctx.lineWidth = 1;
    }
}
