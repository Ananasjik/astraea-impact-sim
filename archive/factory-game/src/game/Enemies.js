class Enemy {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.speed = 50;
        this.radius = 10;
        this.health = 20;
    }

    update(dtSec, player, buildings, world) {
        // Simple AI: Move towards player
        const dx = player.x - this.x;
        const dy = player.y - this.y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        
        if (dist > this.radius + player.radius) {
            this.x += (dx / dist) * this.speed * dtSec;
            this.y += (dy / dist) * this.speed * dtSec;
        } else {
            // Attack player
            player.health -= 5 * dtSec;
            updateHealthBar(player.health, player.maxHealth);
        }
    }
}

class Enemies {
    constructor() {
        this.list = [];
        this.spawnTimer = 0;
        this.spawnInterval = 5000; // spawn every 5 seconds
    }

    update(dt, player, buildings, world) {
        const dtSec = dt / 1000;
        
        // Spawning logic (spawn outside screen)
        this.spawnTimer += dt;
        if (this.spawnTimer >= this.spawnInterval) {
            this.spawnTimer = 0;
            
            // Pick a random angle and spawn at a distance
            const angle = Math.random() * Math.PI * 2;
            const dist = 600; // pixels away from player
            const sx = player.x + Math.cos(angle) * dist;
            const sy = player.y + Math.sin(angle) * dist;
            
            this.list.push(new Enemy(sx, sy));
            
            // Decrease spawn interval over time (difficulty)
            this.spawnInterval = Math.max(1000, this.spawnInterval - 50);
        }

        // Update enemies
        for (let i = this.list.length - 1; i >= 0; i--) {
            const e = this.list[i];
            e.update(dtSec, player, buildings, world);
            
            if (e.health <= 0) {
                this.list.splice(i, 1);
            }
        }
    }
}

function updateHealthBar(health, maxHealth) {
    const bar = document.getElementById('health-bar');
    if (bar) {
        const percent = Math.max(0, (health / maxHealth) * 100);
        bar.style.width = `${percent}%`;
    }
}
