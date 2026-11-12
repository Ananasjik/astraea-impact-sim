class Building {
    constructor(x, y, type) {
        this.x = x;
        this.y = y;
        this.type = type;
        this.isDestroyed = false;
        this.health = 100;
        this.maxHealth = 100;
    }

    update(dt, world) {
        // Override in subclasses
    }
    
    takeDamage(amount) {
        this.health -= amount;
        if (this.health <= 0) {
            this.isDestroyed = true;
        }
    }
}
