mod types;
use std::fs::File;
use std::io::Error;
use rand::Rng;
use types::{Camera, Color, Dielectric, HittableList, Lambertian, Metal, Sphere, Vec3};

fn main() -> Result<(), Error> {
    let mut rng = rand::thread_rng();
    
    //image
    let mut image = File::create("output.ppm")?;
    let aspect_ratio = 16.0 / 9.0;
    let image_w = 1920;
    let samples_per_pixel = 500;
    let max_depth = 50;
    let v_fov = 20.0;
    let lookfrom = Vec3::new(13.0, 2.0, 3.0);
    let lookat = Vec3::new(0.0, 0.0, 0.0);
    let vup = Vec3::new(0.0, 1.0, 0.0);
    let defocus_angle = 0.6;
    let focus_dist = 10.0;

    //world
    let mut world = HittableList::new();
    let material_ground = Lambertian::new(Color::new(0.5, 0.5, 0.5));
    world.add_o(Sphere::new_o(Vec3::new(0.0, -1000.0, 0.0), 1000.0, material_ground));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat: f64 = rng.gen();
            let center = Vec3::new((a as f64) + (0.9 * rng.gen::<f64>()), 0.2, (b as f64) + (0.9 * rng.gen::<f64>()));

            if (center - Vec3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                
                if choose_mat < 0.8 {
                    let albedo = Color::new(rng.gen(), rng.gen(), rng.gen());
                    let sphere_material = Lambertian::new(albedo);
                    world.add_o(Sphere::new_o(center, 0.2, sphere_material))
                } else if choose_mat < 0.95 {
                    let albedo = Color::new(rng.gen_range(0.5..1.0), rng.gen_range(0.5..1.0), rng.gen_range(0.5..1.0));
                    let fuzz = rng.gen_range(0.0..0.5);
                    let sphere_material = Metal::new(albedo, fuzz);
                    world.add_o(Sphere::new_o(center, 0.2, sphere_material))
                } else {
                    let sphere_material = Dielectric::new(1.5);
                    world.add_o(Sphere::new_o(center, 0.2, sphere_material))
                }
            }
        }
    }

    let mat1 = Dielectric::new(1.5);
    world.add_o(Sphere::new_o(Vec3::new(0.0, 1.0, 0.0), 1.0, mat1));

    let mat2 = Lambertian::new(Color::new(0.4, 0.2, 0.1));
    world.add_o(Sphere::new_o(Vec3::new(-4.0, 1.0, 0.0), 1.0, mat2));

    let mat3 = Metal::new(Color::new(0.7, 0.6, 0.5), 0.0);
    world.add_o(Sphere::new_o(Vec3::new(4.0, 1.0, 0.0), 1.0, mat3));


    //render
    let cam = Camera::new(image_w, aspect_ratio, samples_per_pixel, max_depth, v_fov, lookfrom, lookat, vup, defocus_angle, focus_dist);
    cam.render(&mut image, &world, &mut rng)?;

    println!("Done!");
    Ok(())
}