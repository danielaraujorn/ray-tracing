#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include "geometry.h"
#define MAXDEPTH 10
#define BACKGROUNDCOLOR Vec3f(0.3, 0.2, 0.3)

struct Light
{
    Light(const Vec3f &p, const float i) : position(p), intensity(i) {}
    Vec3f position;
    float intensity;
};

struct Material
{
    Material(const float r, const Vec4f &a, const Vec3f &color, const float spec) : refractive_index(r), albedo(a), diffuse_color(color), specular_exponent(spec) {}
    Material() : refractive_index(1), albedo(1, 0, 0, 0), diffuse_color(), specular_exponent() {}
    float refractive_index;
    Vec4f albedo;
    Vec3f diffuse_color;
    float specular_exponent;
};

struct Sphere
{
    Vec3f center;
    float radius;
    Material material;

    Sphere(const Vec3f &c, const float r, const Material &m) : center(c), radius(r), material(m) {}

    bool rayIntersect(const Vec3f &orig, const Vec3f &dir, float &t0) const
    {
        Vec3f oc = center - orig;
        float tca = oc * dir;
        float d2 = oc * oc - tca * tca;
        if (d2 > radius * radius)
            return false;
        float thc = sqrtf(radius * radius - d2);
        t0 = tca - thc;
        float t1 = tca + thc;
        if (t0 < 0)
            t0 = t1;
        return t0 >= 0;
    }
};

Vec3f reflect(const Vec3f &I, const Vec3f &N)
{
    return I - N * 2.f * (I * N);
}

Vec3f refract(const Vec3f &I, const Vec3f &N, const float eta_t, const float eta_i = 1.f)
{ // Snell's law
    float cosi = -std::max(-1.f, std::min(1.f, I * N));
    if (cosi < 0)
        return refract(I, -N, eta_i, eta_t); // if the ray comes from the inside the object, swap the air and the media
    float eta = eta_i / eta_t;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? Vec3f(1, 0, 0) : I * eta + N * (eta * cosi - sqrtf(k)); // k<0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
}

bool sceneIntersect(const Vec3f &orig, const Vec3f &dir, const std::vector<Sphere> &spheres, Vec3f &hit, Vec3f &N, Material &material)
{
    float spheres_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < spheres.size(); i++)
    {
        float dist_i;
        if (spheres[i].rayIntersect(orig, dir, dist_i) && dist_i < spheres_dist)
        {
            spheres_dist = dist_i;
            hit = orig + dir * dist_i;
            N = (hit - spheres[i].center).normalize();
            material = spheres[i].material;
        }
    }

    float checkerboard_dist = std::numeric_limits<float>::max();
    if (fabs(dir.y) > 1e-3)
    {
        float distance = -(orig.y + 5) / dir.y;
        Vec3f pt = orig + dir * distance;
        if (distance > 0 && fabs(pt.x) < 10 && pt.z < -10 && pt.z > -30 && distance < spheres_dist)
        {
            checkerboard_dist = distance;
            hit = pt;
            N = Vec3f(0, 1, 0);
            material.diffuse_color = (int(.5 * hit.x + 1000) + int(.5 * hit.z)) & 1 ? Vec3f(.3, .3, .3) : Vec3f(.1, .1, .1);
        }
    }
    return std::min(spheres_dist, checkerboard_dist) < 1000;
}

Vec3f rayCaster(const Vec3f &orig, const Vec3f &dir, const std::vector<Sphere> &spheres, const std::vector<Light> &lights, size_t depth = 0)
{
    Vec3f point, N;
    Material material;

    if (depth > MAXDEPTH || !sceneIntersect(orig, dir, spheres, point, N, material))
    {
        return BACKGROUNDCOLOR;
    }

    Vec3f reflectDir = reflect(dir, N).normalize();
    Vec3f refractDir = refract(dir, N, material.refractive_index).normalize();
    Vec3f reflectOrig = reflectDir * N < 0 ? point - N * 1e-3 : point + N * 1e-3; // offset the original point to avoid occlusion by the object itself
    Vec3f refractOrig = refractDir * N < 0 ? point - N * 1e-3 : point + N * 1e-3;
    Vec3f reflectColor = rayCaster(reflectOrig, reflectDir, spheres, lights, depth + 1);
    Vec3f refractColor = rayCaster(refractOrig, refractDir, spheres, lights, depth + 1);

    float diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (size_t i = 0; i < lights.size(); i++)
    {
        Vec3f light_dir = (lights[i].position - point).normalize();
        float light_distance = (lights[i].position - point).norm();

        Vec3f shadow_orig = light_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3; // checking if the point lies in the shadow of the lights[i]
        Vec3f shadow_pt, shadow_N;
        Material tmpmaterial;
        if (sceneIntersect(shadow_orig, light_dir, spheres, shadow_pt, shadow_N, tmpmaterial) && (shadow_pt - shadow_orig).norm() < light_distance)
            continue;

        diffuse_light_intensity += lights[i].intensity * std::max(0.f, light_dir * N);
        specular_light_intensity += powf(std::max(0.f, -reflect(-light_dir, N) * dir), material.specular_exponent) * lights[i].intensity;
    }
    return material.diffuse_color * diffuse_light_intensity * material.albedo[0] + Vec3f(1., 1., 1.) * specular_light_intensity * material.albedo[1] + reflectColor * material.albedo[2] + refractColor * material.albedo[3];
}

void render(const std::vector<Sphere> &spheres, const std::vector<Light> &lights)
{
    const int width = 1024;
    const int height = 768;
    const float fov = M_PI / 3.;
    std::vector<Vec3f> framebuffer(width * height);

#pragma omp parallel for
    for (size_t j = 0; j < height; j++)
    { // actual rendering loop
        for (size_t i = 0; i < width; i++)
        {
            float dir_x = (i + 0.5) - width / 2.;
            float dir_y = -(j + 0.5) + height / 2.; // this flips the image at the same time
            float dir_z = -height / (2. * tan(fov / 2.));
            framebuffer[i + j * width] = rayCaster(Vec3f(0, 0, 0), Vec3f(dir_x, dir_y, dir_z).normalize(), spheres, lights);
        }
    }

    std::ofstream ofs; // save the framebuffer to file
    ofs.open("./out.ppm", std::ios::binary);
    ofs << "P6\n"
        << width << " " << height << "\n255\n";
    for (size_t i = 0; i < height * width; ++i)
    {
        Vec3f &c = framebuffer[i];
        float max = std::max(c[0], std::max(c[1], c[2]));
        if (max > 1)
            c = c * (1. / max);
        for (size_t j = 0; j < 3; j++)
        {
            ofs << (char)(255 * std::max(0.f, std::min(1.f, framebuffer[i][j])));
        }
    }
    ofs.close();
}

int main()
{
    Material glass(1.01, Vec4f(0.0, 0.5, 0.1, 0.8), Vec3f(0.6, 0.8, 0.7), 125.);
    Material rubber(1.0, Vec4f(0.9, 0.1, 0.0, 0.0), Vec3f(0.4, 0.1, 0.3), 10.);
    Material rubber2(1.0, Vec4f(0.9, 0.1, 0.4, 0.0), Vec3f(0.3, 0.1, 0.4), 10.);
    Material mirror(1.0, Vec4f(0.0, 10.0, 0.8, 0.0), Vec3f(1.0, 1.0, 1.0), 1425.);

    std::vector<Sphere> spheres;
    spheres.push_back(Sphere(Vec3f(1.0, -1.5, -12), 3, glass));
    spheres.push_back(Sphere(Vec3f(-3.5, 1.5, -18), 2, rubber));
    spheres.push_back(Sphere(Vec3f(3.5, -1.5, -24), 2, rubber2));
    spheres.push_back(Sphere(Vec3f(7, 5, -18), 4, mirror));

    std::vector<Light> lights;
    lights.push_back(Light(Vec3f(-10, 10, 20), 1.4));
    lights.push_back(Light(Vec3f(-30, -30, 20), 1.2));
    lights.push_back(Light(Vec3f(-20, 20, 20), 1.5));
    lights.push_back(Light(Vec3f(30, 50, -25), 0.8));
    lights.push_back(Light(Vec3f(30, 20, 30), 3.0));

    render(spheres, lights);

    return 0;
}
