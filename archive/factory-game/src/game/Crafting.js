const RECIPES = [
    {
        output: { item: 'miner', count: 1 },
        inputs: [{ item: 'iron_plate', count: 5 }, { item: 'copper_wire', count: 2 }]
    },
    {
        output: { item: 'conveyor', count: 2 },
        inputs: [{ item: 'iron_plate', count: 1 }]
    },
    {
        output: { item: 'smelter', count: 1 },
        inputs: [{ item: 'stone', count: 10 }] // Wait, we don't have stone mining easily setup... let's just use iron for now.
    },
    {
        output: { item: 'iron_plate', count: 1 },
        inputs: [{ item: 'iron', count: 2 }, { item: 'coal', count: 1 }] // Hand crafting version
    },
    {
        output: { item: 'copper_wire', count: 2 },
        inputs: [{ item: 'copper', count: 1 }]
    },
    {
        output: { item: 'turret', count: 1 },
        inputs: [{ item: 'iron_plate', count: 10 }, { item: 'copper_wire', count: 5 }]
    }
];

class CraftingSystem {
    static canCraft(recipe, inventory) {
        for (const input of recipe.inputs) {
            if (!inventory.hasItem(input.item, input.count)) {
                return false;
            }
        }
        return true;
    }

    static craft(recipe, inventory) {
        if (!this.canCraft(recipe, inventory)) return false;
        
        // Consume inputs
        for (const input of recipe.inputs) {
            inventory.removeItem(input.item, input.count);
        }
        
        // Add output
        inventory.addItem(recipe.output.item, recipe.output.count);
        return true;
    }
}
