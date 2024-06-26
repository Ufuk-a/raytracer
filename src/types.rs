use core::f64;
use std::{io::{Error, Write}, ops::{Add, Deref, DerefMut, Div, Mul, Neg, Sub}, rc::Rc, sync::Arc};
use rand::{rngs::ThreadRng, Rng};

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3{x, y, z}
    }
    
    pub fn length(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }

    pub fn length_s(&self) -> f64 {
        self.x.powi(2) + self.y.powi(2) + self.z.powi(2)
    }

    pub fn dot(&self, rhs: Vec3) -> f64 {
        (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z)
    }

    pub fn cross(&self, rhs: Vec3) -> Vec3 {
        Vec3{x: self.y * rhs.z - self.z * rhs.y,
             y: self.z * rhs.x - self.x * rhs.z,
             z: self.x * rhs.y - self.y * rhs.x }
    }

    pub fn normalize(&self) -> Vec3 {
        self/self.length()
    }

    pub fn random(min: Option<f64>, max: Option<f64>, rng: &mut ThreadRng) -> Self {
        let min = min.unwrap_or(0.0);
        let max = max.unwrap_or(1.0);

        Vec3{x: rng.gen_range(min..max), y: rng.gen_range(min..max), z: rng.gen_range(min..max)}
    }

    pub fn random_in_unit_sphere(rng: &mut ThreadRng) -> Self {
        loop {
            let p = Vec3::random(Some(-1.0),Some(1.0), rng);
            if p.length_s() < 1.0 {
                return p;
            }
        }
    }

    pub fn random_on_unit_disk(rng: &mut ThreadRng) -> Vec3 {
        loop {
            let p = Vec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0);
            if p.length_s() < 1.0 {
                return p;
            }
        }
    }

    pub fn random_unit_vector(rng: &mut ThreadRng) -> Self {
        loop {
            let p = Vec3::random(Some(-1.0),Some(1.0), rng);
            if p.length_s() < 1.0 {
                return p.normalize();
            }
        }
    }

    pub fn random_on_hemisphere(normal: Vec3, rng: &mut ThreadRng) -> Self {
        let p = Vec3::random_unit_vector(rng);

        if p.dot(normal) > 0.0 {
            return p;
        } else {
            return -p;
        }
    }

    pub fn reflect(&self, normal: Vec3) -> Self {
        self - (normal * self.dot(normal) * 2.0)
    }

    pub fn refract(&self, normal: Vec3, ref_index: f64) -> Self {
        let cos_theta = (-self).dot(normal).min(1.0);
        let r_out_perp = (self + (normal * cos_theta)) * ref_index;
        let r_out_par = normal * -((1.0 - r_out_perp.length_s()).abs().sqrt());
        r_out_perp + r_out_par
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        Vec3{x: -self.x, y: -self.y, z: -self.z}
    }
}

impl Neg for &Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        Vec3{x: -self.x, y: -self.y, z: -self.z}
    }
}

impl Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Self::Output {
        Vec3{x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z}
    }
}

impl Add<Vec3> for &Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Self::Output {
        Vec3{x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z}
    }
}

impl Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3{x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z}
    }
}

impl Sub<Vec3> for &Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3{x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z}
    }
}

impl Mul<Vec3> for Vec3{
    type Output = Vec3; 

    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3{x: self.x * rhs.x, y: self.y * rhs.y, z: self.z * rhs.z}
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f64) -> Self::Output {
        Vec3{x: self.x * rhs, y: self.y * rhs, z: self.z * rhs}
    }
}

impl Mul<Vec3> for &Vec3{
    type Output = Vec3; 

    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3{x: self.x * rhs.x, y: self.y * rhs.y, z: self.z * rhs.z}
    }
}

impl Mul<f64> for &Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f64) -> Self::Output {
        Vec3{x: self.x * rhs, y: self.y * rhs, z: self.z * rhs}
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Self::Output {
        Vec3{x: self.x / rhs, y: self.y / rhs, z: self.z / rhs}
    }
}

impl Div<f64> for &Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Self::Output {
        Vec3{x: self.x / rhs, y:  self.y / rhs, z: self.z / rhs}
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Color {
    pub vec: Vec3,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64) -> Color {
        Color {
            vec: Vec3::new(r, g, b),
        }
    }
}

impl Mul<f64> for Color {
    type Output = Color;

    fn mul(self, rhs: f64) -> Self::Output {
        Color{vec: Vec3{x: self.x * rhs, y: self.y * rhs, z: self.z * rhs}}
    }
}

impl Mul<Color> for Color {
    type Output = Color;

    fn mul(self, rhs: Color) -> Self::Output {
        Color{vec: Vec3{x: self.x * rhs.x, y: self.y * rhs.y, z: self.z * rhs.z}}
    }
}

impl Add<Color> for Color {
    type Output = Color;

    fn add(self, rhs: Color) -> Self::Output {
        Color{vec: Vec3{x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z}}
    }
}

impl Deref for Color {
    type Target = Vec3;

    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl DerefMut for Color {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Ray {
    pub orig: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn new(orig: Vec3, dir: Vec3) -> Ray{
        Ray{orig, dir}
    }
    
    pub fn at(&self, t: f64) -> Vec3{
        self.orig + (self.dir * t)
    }
}

#[derive(Clone)]
pub struct HitRecord {
    pub p: Vec3,
    pub normal: Vec3,
    pub t: f64,
    pub front_face: bool,
    pub mat: Rc<dyn Material>,
}

impl Default for HitRecord {
    fn default() -> Self {
        Self {
            p: Vec3::new(0f64, 0f64, 0f64),
            normal: Vec3::new(0f64, 0f64, 0f64),
            t: 0.0,
            front_face: false,
            mat: Rc::new(Lambertian{albedo: Color::new(0.0, 0.0, 0.0)}),
        }
    }
}

impl HitRecord {
    fn set_face_normal(&mut self, r: Ray, outward_normal: Vec3) {     
        self.front_face = r.dir.dot(outward_normal) < 0.0;
        if self.front_face {
            self.normal = outward_normal;
        } else {
            self.normal = -outward_normal;
        }
    }
}

pub trait Hittable {
    fn hit(&self, r: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool;
}

pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
    pub mat: Rc<dyn Material>,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f64, mat: Rc<dyn Material>) -> Self {
        Sphere{center: center, radius: radius, mat:mat}
    }

    pub fn new_o(center: Vec3, radius: f64, mat: impl Material + 'static) -> Self {
        Sphere::new(center, radius, Rc::new(mat))
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        let oc = self.center - r.orig;
        let a = r.dir.length_s();
        let h = r.dir.dot(oc);
        let c = oc.length_s() - self.radius.powi(2);
        
        let d = h.powi(2) - a*c;
        if d < 0.0 {
            return false;
        }

        let sqrtd = d.sqrt();

        let mut root = (h - sqrtd) / a;
        if !ray_t.surrounds(root) {
            root = (h + sqrtd) / a;
            if !ray_t.surrounds(root) {
                return false;
            }
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        let outward_normal = (rec.p - self.center) / self.radius;
        rec.set_face_normal(*r, outward_normal);
        rec.mat = Rc::clone(&self.mat);

        true
    }
}

#[derive(Clone)]
pub struct HittableList {
    objects: Vec<Arc<dyn Hittable>>,
}

impl HittableList {
    pub fn new() -> Self {
        HittableList{objects: Vec::new()}
    }

    pub fn with_object(object: Arc<dyn Hittable>) -> Self {
        let mut list = HittableList::new();
        list.add(object);
        list
    }

    pub fn clear(&mut self) {
        self.objects.clear();
    }

    pub fn add(&mut self, object: Arc<dyn Hittable>) {
        self.objects.push(object);
    }

    pub fn add_o(&mut self, object: impl Hittable + 'static) {
        self.add(Arc::new(object));
    }
}

impl Hittable for HittableList {
    fn hit(&self, r: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        let mut temp_rec = HitRecord::default();
        let mut hit_anything = false;
        let mut closest = ray_t.max;

        for object in &self.objects {
            if object.hit(r, Interval{min: ray_t.min, max: closest}, &mut temp_rec) {
                hit_anything = true;
                closest = temp_rec.t;
                *rec = temp_rec.clone();
            }
        }

        hit_anything
    }
}

pub struct Interval {
    pub min: f64,
    pub max: f64,
}

impl Interval {
    pub fn new(min: f64, max: f64) -> Self {
        Interval{min:min, max:max}
    }

    pub fn size(&self) -> f64 {
        self.max - self.min
    }

    pub fn contains(&self, x: f64) -> bool {
        self.min <= x && x <= self.max
    }

    pub fn surrounds(&self, x: f64) -> bool {
        self.min < x && x < self.max
    }

    pub fn clamp(&self, x: f64) -> f64 {
        if x < self.min {
            return self.min;
        }
        if x > self.max {
            return self.max;
        }
        x
    }
    
    const EMPTY: Interval = Interval { min: f64::INFINITY, max: f64::NEG_INFINITY };
    const UNIVERSE: Interval = Interval { min: f64::NEG_INFINITY, max: f64::INFINITY };
}

impl Default for Interval {
    fn default() -> Self {
        Self{min: f64::NEG_INFINITY, max: f64::INFINITY}
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Camera {
    pub aspect_ratio: f64,
    pub image_w: i32,
    pub image_h: i32,
    pub center: Vec3,
    pub pixel00_l: Vec3,
    pub pixel_delta_u: Vec3,
    pub pixel_delta_v: Vec3,
    pub samples_per_pixel: i32,
    pub pixel_samples_scale: f64,
    pub max_depth: i32,
    pub v_fov: f64,
    pub lookfrom: Vec3,
    pub lookat: Vec3,
    pub vup: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
    pub defocus_angle: f64,
    pub focus_dist: f64,
    pub defocus_disk_u: Vec3,
    pub defocus_disk_v: Vec3,
}

impl Camera {
    pub fn new(image_w: i32, aspect_ratio: f64, samples_per_pixel: i32, max_depth: i32, v_fov: f64, lookfrom: Vec3, lookat: Vec3, vup: Vec3, defocus_angle: f64, focus_dist:f64) -> Self {
    let mut image_h = (image_w as f64 /aspect_ratio) as i32;
    if image_h < 1 {
        image_h = 1;
    }

    let center = lookfrom;
    let pixel_samples_scale = 1.0 / (samples_per_pixel as f64);
    
    let theta = v_fov.to_radians();
    let h = (theta / 2.0).tan();
    let viewport_h = 2.0 * h * focus_dist;
    let viewport_w = viewport_h * (image_w as f64/image_h as f64);
    let camera_c = Vec3::new(0f64, 0f64, 0f64);

    let w = (lookfrom - lookat).normalize();
    let u = vup.cross(w).normalize();
    let v = w.cross(u);

    let viewport_u = u * viewport_w;
    let viewport_v = -v * viewport_h;

    let pixel_delta_u = viewport_u / image_w as f64;
    let pixel_delta_v = viewport_v / image_h as f64;

    let viewport_upper_left = center - (w * focus_dist) - viewport_u/2.0 - viewport_v/2.0;
    let pixel00_l = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5;

    let defocus_radius = focus_dist * (defocus_angle / 2.0).to_radians().tan();
    let defocus_disk_u = u * defocus_radius;
    let defocus_disk_v = v * defocus_radius;

    Camera{aspect_ratio, image_w, image_h, center, pixel00_l, pixel_delta_u, pixel_delta_v, samples_per_pixel, pixel_samples_scale, max_depth, v_fov, lookfrom, lookat, vup, u, v, w, defocus_angle, focus_dist, defocus_disk_u, defocus_disk_v}
    }

    pub fn ray_color(&self, r: &Ray, world: &impl Hittable, depth: i32, rng: &mut ThreadRng) -> Color {
    
        if depth <= 0 {
            return Color::new(0.0, 0.0, 0.0);
        }
        
        let mut rec = HitRecord::default();
        if world.hit(r, Interval{min: 0.001, max: f64::INFINITY}, &mut rec) {
            let mut scattered = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
            let mut attenuation = Color::new(0.0, 0.0, 0.0);            
    
            if rec.mat.scatter(r, &rec, &mut attenuation, &mut scattered, rng) {
                let color = self.ray_color(&scattered, world, depth - 1, rng);
                return attenuation * color;
            }
            return Color::new(0.0, 0.0, 0.0);
        }
    
        let a = (r.dir.normalize().y + 1.0) * 0.5;
        let color = (Color::new(1.0, 1.0, 1.0) * (1.0 - a)) + (Color::new(0.5, 0.7, 1.0) * a);
        color
    }

    pub fn write_color(&self, out: &mut dyn Write, pixel_color: Color) -> Result<(), Error> {
        let r = pixel_color.x;
        let g = pixel_color.y;
        let b = pixel_color.z;
    
        let intensity = Interval::new(0.000, 0.999);
        let rb = (255.999 * intensity.clamp(r.sqrt())) as u8;
        let gb = (255.999 * intensity.clamp(g.sqrt())) as u8;
        let bb = (255.999 * intensity.clamp(b.sqrt())) as u8;
    
        writeln!(out, "{} {} {}", rb, gb, bb)
    }
    pub fn render(&self, image: &mut dyn Write, world: &impl Hittable, rng: &mut ThreadRng) -> Result<(), Error> {
        image.write_all(format!("P3\n{} {}\n255\n", self.image_w, self.image_h).as_bytes())?;
    
        for y in 0..self.image_h {
            println!("Scanlines Remaining: {}", (self.image_h - y));
            for x in 0..self.image_w {
                let mut pixel_color = Color::new(0.0, 0.0, 0.0);
                for _sample in 0..self.samples_per_pixel {
                    let r = self.get_ray(x, y, rng);
                    pixel_color = pixel_color + self.ray_color(&r, world, self.max_depth, rng);
                }
                
                pixel_color = pixel_color * self.pixel_samples_scale;
                self.write_color(image, pixel_color)?;
            }
        }
    
        Ok(())
    }

    pub fn get_ray(&self, x: i32, y: i32, rng: &mut ThreadRng) -> Ray {
        let offset = Vec3::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5, 0.0);
        let pixel_sample = self.pixel00_l + (self.pixel_delta_u * (offset.x + x as f64)) + (self.pixel_delta_v * (offset.y + y as f64));
        let ray_orig = if self.defocus_angle <= 0.0 {
            self.center
        } else {
            self.defocus_disk_sample(rng)
        };
        let ray_dir = (pixel_sample - ray_orig).normalize();

        Ray::new(ray_orig, ray_dir)
    }

    pub fn defocus_disk_sample(&self, rng: &mut ThreadRng) -> Vec3 {
        let p = Vec3::random_on_unit_disk(rng);
        self.center + (self.defocus_disk_u * p.x) + (self.defocus_disk_v * p.y)
    }
}

pub trait Material {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, attenuation: &mut Color, scattered: &mut Ray, rng: &mut ThreadRng) -> bool; 
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Lambertian {
    pub albedo: Color,
}

impl Lambertian {
    pub fn new(albedo: Color) -> Self {
        Lambertian{albedo}
    }
}

impl Material for Lambertian {
    fn scatter(&self, _r_in: &Ray, rec: &HitRecord, attenuation: &mut Color, scattered: &mut Ray, rng: &mut ThreadRng) -> bool {
        let mut scatter_direction = rec.normal + Vec3::random_unit_vector(rng);
        if scatter_direction.x < 1e-10 && scatter_direction.y < 1e-10 && scatter_direction.z < 1e-10 {
            scatter_direction = rec.normal;
        }
        *scattered = Ray::new(rec.p, scatter_direction);
        *attenuation = self.albedo;
        true
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Metal {
    pub albedo: Color,
    pub fuzz: f64,
}

impl Metal {
    pub fn new(albedo: Color, fuzz: f64) -> Self {
        Metal{albedo, fuzz}
    }
}

impl Material for Metal {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, attenuation: &mut Color, scattered: &mut Ray, rng: &mut ThreadRng) -> bool {
        let reflected = r_in.dir.normalize().reflect(rec.normal);
        *scattered = Ray::new(rec.p, reflected + Vec3::random_unit_vector(rng) * self.fuzz);
        *attenuation = self.albedo;
        let scatter_success = scattered.dir.dot(rec.normal) > 0.0;
        scatter_success
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Dielectric {
    pub ref_index: f64,
}

impl Dielectric {
    pub fn new(ref_index: f64) -> Self{
        Dielectric{ref_index}
    }
}

impl Material for Dielectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, attenuation: &mut Color, scattered: &mut Ray, rng: &mut ThreadRng) -> bool {
        *attenuation = Color::new(1.0, 1.0, 1.0);
        let ri = if rec.front_face {
            1.0 / self.ref_index
        } else {
            self.ref_index
        };

        let unit_dir = r_in.dir.normalize();
        let cos_theta = -unit_dir.dot(rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta.powi(2)).sqrt();

        let cannot_refract = ri * sin_theta > 1.0;
        let direction: Vec3;
        let rf = rng.gen_range(0.0..1.0);

        if cannot_refract || self.reflactence(cos_theta, ri) > rf {
            direction = unit_dir.reflect(rec.normal);
        } else {
            direction = unit_dir.refract(rec.normal, ri);
        }

        *scattered = Ray::new(rec.p, direction);
        true
    }
}

impl Dielectric {
    pub fn reflactence(&self, cosine: f64, ref_index: f64) -> f64 {
        let mut r0 = (1.0 - ref_index) / (1.0 + ref_index);
        r0 = r0.powi(2);
        r0 + ((1.0 - r0) * ((1.0 - cosine).powi(5)))
    }
}