class Player {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.speed = 200; // pixels per second
        this.radius = 12;
        this.angle = 0;
        this.health = 100;
        this.maxHealth = 100;
        
        // this.inventory = new Inventory(20);
    }

    update(dt, input, world) {
        let dx = 0;
        let dy = 0;

        if (input.isKeyDown('KeyW')) dy -= 1;
        if (input.isKeyDown('KeyS')) dy += 1;
        if (input.isKeyDown('KeyA')) dx -= 1;
        if (input.isKeyDown('KeyD')) dx += 1;

        if (dx !== 0 || dy !== 0) {
            // Normalize
            const len = Math.sqrt(dx*dx + dy*dy);
            dx /= len;
            dy /= len;
            
            this.angle = Math.atan2(dy, dx);
            
            // Convert dt to seconds
            const dtSec = dt / 1000;
            this.x += dx * this.speed * dtSec;
            this.y += dy * this.speed * dtSec;
        }

        // Basic collision with water/world edges could go here
        
        // Manual mining
        if (input.isMouseJustPressed(2)) { // Right click to mine manually
            const worldPos = window.game.camera.screenToWorld(input.mouseX, input.mouseY);
            const tileX = Math.floor(worldPos.x / TILE_SIZE);
            const tileY = Math.floor(worldPos.y / TILE_SIZE);
            const tile = world.getTile(tileX, tileY);
            
            if (tile.ore) {
                // ADD TO INVENTORY (To be implemented)
                console.log(`Mined ${tile.ore} at ${tileX}, ${tileY}`);
                // tile.ore = null; // Maybe ores are infinite for now, or deplete? Let's make them deplete slowly or infinite for prototype.
            } else if (tile.building) {
                console.log(`Mined building ${tile.building.type}`);
                tile.building.isDestroyed = true;
                tile.building = null;
            }
        }
    }
}
