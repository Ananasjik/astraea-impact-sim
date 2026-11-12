// Main Game Loop and Initialization
const TILE_SIZE = 32;

class Game {
    constructor() {
        this.canvas = document.getElementById('game-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.lastTime = 0;
        this.accumulator = 0;
        this.timestep = 1000 / 60; // 60 FPS
        
        this.setupCanvas();
        window.addEventListener('resize', () => this.setupCanvas());
        
        // Systems
        this.input = new Input(this.canvas);
        this.camera = new Camera(this.canvas.width, this.canvas.height);
        
        // Game State
        this.world = new World();
        this.player = new Player(0, 0);
        this.enemies = new Enemies();
        this.buildings = []; // All placed buildings
        
        this.renderer = new Renderer(this.ctx, this.camera);
        
        // UI State
        this.ui = {
            inventoryOpen: false,
            craftingOpen: false,
            hotbar: ['miner', 'conveyor', 'smelter', 'turret'], // Available buildings for prototype
            selectedHotbarSlot: 0,
            buildDirection: 0 // 0, 1, 2, 3
        };
        
        this.initUI();
    }
    
    setupCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        if (this.camera) {
            this.camera.resize(this.canvas.width, this.canvas.height);
        }
    }
    
    initUI() {
        // Toggle inventory with E
        window.addEventListener('keydown', (e) => {
            if (e.key === 'e' || e.key === 'E') {
                this.ui.inventoryOpen = !this.ui.inventoryOpen;
                document.getElementById('inventory-panel').classList.toggle('hidden', !this.ui.inventoryOpen);
            }
            if (e.key === 'c' || e.key === 'C') {
                this.ui.craftingOpen = !this.ui.craftingOpen;
                document.getElementById('crafting-panel').classList.toggle('hidden', !this.ui.craftingOpen);
            }
            if (e.key === 'r' || e.key === 'R') {
                this.ui.buildDirection = (this.ui.buildDirection + 1) % 4;
            }
            if (e.key >= '1' && e.key <= '4') {
                this.ui.selectedHotbarSlot = parseInt(e.key) - 1;
                this.renderUI();
            }
        });
        
        this.renderUI();
    }
    
    renderUI() {
        const hotbarEl = document.getElementById('hotbar');
        hotbarEl.innerHTML = '';
        this.ui.hotbar.forEach((item, index) => {
            const el = document.createElement('div');
            el.className = 'slot' + (index === this.ui.selectedHotbarSlot ? ' selected' : '');
            
            const def = ITEMS[item];
            if (def) {
                const colorBox = document.createElement('div');
                colorBox.style.width = '20px';
                colorBox.style.height = '20px';
                colorBox.style.backgroundColor = def.icon;
                el.appendChild(colorBox);
                
                const num = document.createElement('span');
                num.className = 'count';
                num.innerText = index + 1;
                el.appendChild(num);
            }
            hotbarEl.appendChild(el);
        });
    }

    start() {
        requestAnimationFrame((time) => this.loop(time));
    }

    loop(time) {
        const deltaTime = time - this.lastTime;
        this.lastTime = time;
        this.accumulator += deltaTime;

        // Fixed timestep update
        while (this.accumulator >= this.timestep) {
            this.update(this.timestep);
            this.accumulator -= this.timestep;
        }

        // Render (can pass accumulator/timestep for interpolation if needed)
        this.render();

        this.input.postUpdate(); // Clear one-frame input states like 'justPressed'
        requestAnimationFrame((t) => this.loop(t));
    }

    update(dt) {
        this.player.update(dt, this.input, this.world);
        this.camera.follow(this.player.x, this.player.y);
        this.world.update(dt, this.player);
        this.enemies.update(dt, this.player, this.buildings, this.world);
        
        for (let i = this.buildings.length - 1; i >= 0; i--) {
            const b = this.buildings[i];
            b.update(dt, this.world);
            if (b.isDestroyed) {
                this.world.setBuilding(b.x, b.y, null);
                this.buildings.splice(i, 1);
            }
        }
        
        // Handle building placement
        if (this.input.isMouseJustPressed(0) && !this.input.isMouseOverUI()) {
            const worldPos = this.camera.screenToWorld(this.input.mouseX, this.input.mouseY);
            const tileX = Math.floor(worldPos.x / TILE_SIZE);
            const tileY = Math.floor(worldPos.y / TILE_SIZE);
            
            if (this.world.canPlaceBuilding(tileX, tileY)) {
                const typeToBuild = this.ui.hotbar[this.ui.selectedHotbarSlot];
                let b = null;
                
                if (typeToBuild === 'miner') b = new Miner(tileX, tileY);
                else if (typeToBuild === 'conveyor') b = new Conveyor(tileX, tileY, this.ui.buildDirection);
                else if (typeToBuild === 'smelter') b = new Smelter(tileX, tileY);
                else if (typeToBuild === 'turret') b = new Turret(tileX, tileY);
                
                if (b) {
                    this.buildings.push(b);
                    this.world.setBuilding(tileX, tileY, b);
                }
            }
        }
    }

    render() {
        this.ctx.fillStyle = '#111'; // Void background color
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.ctx.save();
        this.camera.apply(this.ctx);

        this.renderer.drawWorld(this.world);
        this.renderer.drawBuildings(this.buildings);
        this.renderer.drawEnemies(this.enemies);
        this.renderer.drawPlayer(this.player);
        
        // Draw placement highlight
        const worldPos = this.camera.screenToWorld(this.input.mouseX, this.input.mouseY);
        const tileX = Math.floor(worldPos.x / TILE_SIZE);
        const tileY = Math.floor(worldPos.y / TILE_SIZE);
        this.renderer.drawHighlight(tileX, tileY);

        this.ctx.restore();
    }
}

// Start game when DOM is ready
window.addEventListener('DOMContentLoaded', () => {
    window.game = new Game();
    window.game.start();
});
