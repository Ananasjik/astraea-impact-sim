// Define all items in the game
const ITEMS = {
    'coal': { name: 'Coal', icon: '#222' },
    'iron': { name: 'Iron Ore', icon: '#8a9b9e' },
    'copper': { name: 'Copper Ore', icon: '#c4693b' },
    'iron_plate': { name: 'Iron Plate', icon: '#ccc' },
    'copper_wire': { name: 'Copper Wire', icon: '#e07941' },
    
    // Buildings as items
    'miner': { name: 'Miner', icon: '#aa3333', isBuilding: true, type: 'miner' },
    'conveyor': { name: 'Conveyor', icon: '#ffff00', isBuilding: true, type: 'conveyor' },
    'smelter': { name: 'Smelter', icon: '#ff5500', isBuilding: true, type: 'smelter' },
    'turret': { name: 'Turret', icon: '#4444ff', isBuilding: true, type: 'turret' }
};
