mod types;
use std::{fs::File};
use std::io::Error;
use types::{Camera, Color, Dielectric, HittableList, Lambertian, Metal, Sphere, Vec3};

fn main() -> Result<(), Error> {
    let mut rng = rand::thread_rng();
    
    //image
    let mut image = File::create("output.ppm")?;
    let aspect_ratio = 16.0 / 9.0;
    let image_w = 400;
    let samples_per_pixel = 10;
    let max_depth = 50;
    
    //world
    let mut world = HittableList::new();
    let material_ground = Lambertian::new(Color::new(0.8, 0.8, 0.0));
    let material_center = Lambertian::new(Color::new(0.1, 0.2, 0.5));
    let material_left = Dielectric::new(1.5);
    let material_right = Metal::new(Color::new(0.8, 0.6, 0.2), 1.0);

    world.add_o(Sphere::new_o(Vec3::new(0.0, -100.5, -1.0), 100.0, material_ground));
    world.add_o(Sphere::new_o(Vec3::new(0.0, 0.0, -1.0), 0.5, material_center));
    world.add_o(Sphere::new_o(Vec3::new(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add_o(Sphere::new_o(Vec3::new(1.0, 0.0, -1.0), 0.5, material_right));

    //render
    let cam = Camera::new(image_w, aspect_ratio, samples_per_pixel, max_depth);
    cam.render(&mut image, &world, &mut rng)?;

    println!("Done!");
    Ok(())
}