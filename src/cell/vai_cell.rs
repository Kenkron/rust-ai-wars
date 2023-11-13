use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::math::*;
use bevy::time::common_conditions::*;
use bevy_rapier2d::prelude::*;
use rand::Rng;

pub const NUM_VAI_CELLS: usize = 1000;
pub const NUM_INPUT_NODES_W_BIAS: usize = NUM_INPUT_NODES + 1;
pub const VAI_CELL_SPRITE: &str = "vai-turret.png";

use crate::{
    cell::*,
    food::FoodTree,
    gui::SimStats,
    vain::VaiNet,
    settings::SimSettings,
    trackers::{
        BirthPlace, BirthTs, FitnessScores, LastBulletFired, LastUpdated, NumCellsSpawned,
        OneSecondTimer, PeriodicUpdateInterval,
    },
    *,
};

use super::{
    energy::EnergyMap,
    focus::{FocusedCellNet, FocusedCellStats},
    user::UserControlledCell,
};

#[derive(Component)]
pub struct VaiBrain(pub VaiNet<NUM_INPUT_NODES_W_BIAS, NUM_OUTPUT_NODES, NUM_HIDDEN_NODES>);

pub struct VaiCellPlugin;

impl Plugin for VaiCellPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_vai_cells_system)
            .add_systems(
                Update,
                vai_cell_replication_system.run_if(on_timer(Duration::from_secs_f32(0.5))),
            )
            .add_systems(
                Update,
                spawn_vai_cells.run_if(on_timer(Duration::from_secs_f32(5.0))),
            );
    }
}

fn spawn_vai_cells(
    mut commands: Commands,
    mut cell_id: ResMut<CellId>,
    asset_server: Res<AssetServer>,
    cell_query: Query<(With<Cell>, With<VaiBrain>, Without<UserControlledCell>)>,
) {
    let num_cells = cell_query.iter().len();
    if num_cells > 0 {
        return;
    }

    let mut rng = rand::thread_rng();
    for _ in 0..NUM_VAI_CELLS {
        let x = rng.gen_range(-(W as f32) / 2.0..W as f32 / 2.0);
        let y = rng.gen_range(-(H as f32) / 2.0..H as f32 / 2.0);
        let net = VaiNet::<NUM_INPUT_NODES_W_BIAS, NUM_OUTPUT_NODES, NUM_HIDDEN_NODES>::new();

        cell_id.0 += 1;
        commands.spawn(VaiCellBundle::new(
            x,
            y,
            cell_id.0,
            net,
            VAI_CELL_SPRITE,
            &asset_server,
        ));
    }
}

fn update_vai_cells_system(
    mut commands: Commands,
    one_second_timer: Res<OneSecondTimer>,
    asset_server: Res<AssetServer>,
    food_tree: Res<FoodTree>,
    focused_cell_stats: Res<FocusedCellStats>,
    mut focused_cell_net: ResMut<FocusedCellNet>,
    mut cell_query: Query<
        (
            &Cell,
            &mut Transform,
            &VaiBrain,
            &mut ExternalForce,
            &mut LastUpdated,
            &mut LastBulletFired,
            &mut FitnessScores,
            &PeriodicUpdateInterval,
        ),
        (With<Cell>, Without<UserControlledCell>),
    >,
) {
    for (
        cell,
        mut transform,
        brain,
        mut external_force,
        mut last_updated,
        mut last_bullet_fired,
        mut fitness_scores,
        periodic_update_interval,
    ) in cell_query.iter_mut()
    {
        if last_updated.0.elapsed_within(UPDATE_INTERVAL) {
            continue;
        }
        if one_second_timer
            .0
            .elapsed_within(periodic_update_interval.0)
        {
            continue;
        }

        last_updated.0.set_instant_now();

        let input = get_nn_inputs(&transform, &food_tree);

        // Update brain
        let output = &brain.0.predict(&input.to_vec());
        if focused_cell_stats.id == cell.0 {
            focused_cell_net.0 = output.clone();
        }

        let output = &output[NET_ARCH.len() - 1];

        let fitness = calc_fitness(input, [output[0], output[1], output[2], output[3]]);
        fitness_scores.push(fitness);

        let action = get_nn_cell_action(output);
        perform_cell_action(
            action,
            cell.0,
            &mut last_bullet_fired,
            &mut external_force,
            &mut commands,
            &mut transform,
            &asset_server,
        );
    }
}

fn vai_cell_replication_system(
    mut commands: Commands,
    mut cell_id: ResMut<CellId>,
    energy_map: Res<EnergyMap>,
    stats: Res<SimStats>,
    asset_server: Res<AssetServer>,
    mut cell_query: Query<(&Cell, &VaiBrain, &mut NumCellsSpawned), With<Cell>>,
) {
    let mut num_cells = cell_query.iter().len();
    for (c, brain, mut num_cells_spawned) in cell_query.iter_mut() {
        let mut rng = rand::thread_rng();
        if num_cells >= NUM_VAI_CELLS {
            continue;
        }

        match energy_map.0.get(&c.0) {
            Some((v, _)) => {
                if rng.gen_range(0.0..1.0) >= (v / stats.max_score) {
                    continue;
                }
                // if rng.gen_range(0.0..100.0) >= (birth_ts.0.elapsed() / stats.max_age) * 20.0 {
                //     continue;
                // }
                // if rng.gen_range(0.0..100.0) >= 20.0 {
                //     continue;
                // }

                let x = rng.gen_range(-(W as f32) / 2.0..W as f32 / 2.0);
                let y = rng.gen_range(-(H as f32) / 2.0..H as f32 / 2.0);
                let mut child_net = brain.0.clone();
                child_net.mutate();

                cell_id.0 += 1;
                num_cells += 1;

                num_cells_spawned.0 += 1;
                commands.spawn(VaiCellBundle::new(
                    x,
                    y,
                    cell_id.0,
                    child_net,
                    VAI_CELL_SPRITE,
                    &asset_server,
                ));
            }
            None => {}
        }
    }
}

#[derive(Bundle)]
pub struct VaiCellBundle {
    sprite_bundle: SpriteBundle,
    cell: Cell,
    birth_place: BirthPlace,
    birth_ts: BirthTs,
    last_bullet_fired: LastBulletFired,
    periodic_update_interval: PeriodicUpdateInterval,
    last_updated: LastUpdated,
    rigid_body: RigidBody,
    collider: Collider,
    damping: Damping,
    brain: VaiBrain,
    num_cells_spawned: NumCellsSpawned,
    fitness_score: FitnessScores,
    external_force: ExternalForce,
    collision_groups: CollisionGroups,
}

impl VaiCellBundle {
    pub fn new(
        x: f32,
        y: f32,
        cell_id: u32,
        net: VaiNet<NUM_INPUT_NODES_W_BIAS, NUM_OUTPUT_NODES, NUM_HIDDEN_NODES>,
        sprite_path: &str,
        asset_server: &AssetServer,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let rot = rng.gen_range(0.0..6.0);
        Self {
            sprite_bundle: SpriteBundle {
                transform: Transform::from_xyz(x, y, 1.0)
                    .with_rotation(Quat::from_rotation_z(rot))
                    .with_scale(Vec3::splat(1.5)),
                texture: asset_server.load(sprite_path),
                ..default()
            },
            cell: Cell(cell_id),
            birth_place: BirthPlace(vec2(x, y)),
            birth_ts: BirthTs::default(),
            last_bullet_fired: LastBulletFired::default(),
            periodic_update_interval: PeriodicUpdateInterval(rng.gen_range(0.0..=1.0)),
            last_updated: LastUpdated::default(),
            rigid_body: RigidBody::Dynamic,
            collider: Collider::ball(7.0),
            damping: Damping {
                angular_damping: 2.0,
                linear_damping: 2.0,
            },
            brain: VaiBrain(net),
            num_cells_spawned: NumCellsSpawned(0),
            fitness_score: FitnessScores::new(),
            external_force: ExternalForce {
                force: Vec2::ZERO,
                torque: 0.0,
            },
            collision_groups: CollisionGroups {
                memberships: Group::from_bits_truncate(GRP_CELLS),
                filters: Group::from_bits_truncate(MASK_CELLS),
            },
        }
    }
}