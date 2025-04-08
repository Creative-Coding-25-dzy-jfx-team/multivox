#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include "rammel.h"
#include "input.h"
#include "voxel.h"
#include "mathc.h"
#include "graphics.h"

/* 扩展超维度几何结构 */
#define MAX_DIMENSIONS 5
#define MAX_VERTICES 128  /* 增加到128个顶点 */
#define MAX_EDGES 512    /* 增加边数上限 */
#define M_PI 3.14159265358979323846
typedef struct {
    float coords[MAX_DIMENSIONS];
} hyper_vertex_t;

typedef struct {
    float position[3];
    float velocity[3];
    float size;
    float life;
    float hue;
    uint8_t active;
    uint8_t type;  /* 粒子类型 */
} particle_t;

#define MAX_PARTICLES 600  /* 增加到600个粒子 */
#define TIME_SCALE 0.05f

/* 多种几何体定义 */
#define GEOM_HYPERCUBE 0
#define GEOM_HYPERSPHERE 1
#define GEOM_HYPERSTAR 2
#define GEOM_HYPERSPIRAL 3
#define NUM_GEOM_TYPES 4

static particle_t particles[MAX_PARTICLES];
static hyper_vertex_t vertices[MAX_VERTICES];
static int vertex_pairs[MAX_EDGES][2];
static int num_vertices = 0;
static int num_pairs = 0;
static float time_flow = 0.0f;

static int current_geometry = GEOM_HYPERCUBE;

/* 生成超立方体 */
static void create_hypercube(int dimensions, float size) {
    num_vertices = 0;
    num_pairs = 0;
    
    /* 2^n个顶点的n维超立方体 */
    int vertex_count = 1 << dimensions;
    if (vertex_count > MAX_VERTICES) vertex_count = MAX_VERTICES;
    
    /* 生成顶点 */
    for (int i = 0; i < vertex_count; i++) {
        for (int d = 0; d < dimensions; d++) {
            vertices[num_vertices].coords[d] = ((i & (1 << d)) ? size : -size);
        }
        /* 初始化未使用的维度为0 */
        for (int d = dimensions; d < MAX_DIMENSIONS; d++) {
            vertices[num_vertices].coords[d] = 0.0f;
        }
        num_vertices++;
    }
    
    /* 生成边 */
    for (int i = 0; i < num_vertices && num_pairs < MAX_EDGES; i++) {
        for (int j = i+1; j < num_vertices && num_pairs < MAX_EDGES; j++) {
            /* 计算汉明距离，如果是1则连接顶点 */
            int hamming = 0;
            for (int d = 0; d < dimensions; d++) {
                if ((vertices[i].coords[d] > 0) != (vertices[j].coords[d] > 0)) {
                    hamming++;
                }
            }
            if (hamming == 1) {
                vertex_pairs[num_pairs][0] = i;
                vertex_pairs[num_pairs][1] = j;
                num_pairs++;
            }
        }
    }
}

/* 生成超球体 */
static void create_hypersphere(int points_per_dim, float size) {
    num_vertices = 0;
    num_pairs = 0;
    
    /* 生成球体顶点 - 使用球坐标系统 */
    int total_points = points_per_dim * points_per_dim * 2;
    if (total_points > MAX_VERTICES) total_points = MAX_VERTICES;
    
    /* 黄金角螺旋分布 - 这样可以均匀分布球体表面的点 */
    float phi = M_PI * (3.0f - sqrtf(5.0f)); /* 黄金角 ~2.4 */
    
    for (int i = 0; i < total_points && num_vertices < MAX_VERTICES; i++) {
        float y = 1.0f - (float)i / (total_points - 1) * 2.0f;
        float radius = sqrtf(1.0f - y*y);
        
        float theta = phi * i;
        
        float x = cosf(theta) * radius;
        float z = sinf(theta) * radius;
        
        /* 将球体坐标映射到5D空间 */
        vertices[num_vertices].coords[0] = x * size;
        vertices[num_vertices].coords[1] = y * size;
        vertices[num_vertices].coords[2] = z * size;
        
        /* 添加额外维度的映射 */
        vertices[num_vertices].coords[3] = (sinf(theta * 2.0f) + cosf(y * M_PI)) * size * 0.5f;
        vertices[num_vertices].coords[4] = (cosf(theta * 1.5f) + sinf(y * M_PI * 0.5f)) * size * 0.5f;
        
        num_vertices++;
    }
    
    /* 连接周围的点来形成边 */
    for (int i = 0; i < num_vertices && num_pairs < MAX_EDGES; i++) {
        for (int j = i+1; j < num_vertices && num_pairs < MAX_EDGES; j++) {
            float dist_sq = 0;
            for (int d = 0; d < 3; d++) {
                float diff = vertices[i].coords[d] - vertices[j].coords[d];
                dist_sq += diff * diff;
            }
            
            /* 连接靠近的点 */
            if (dist_sq < (size*size*0.2f)) {
                vertex_pairs[num_pairs][0] = i;
                vertex_pairs[num_pairs][1] = j;
                num_pairs++;
            }
        }
    }
}

/* 生成多维星形 */
static void create_hyperstar(int arms, int points_per_arm, float size) {
    num_vertices = 0;
    num_pairs = 0;
    
    /* 中心点 */
    for (int d = 0; d < MAX_DIMENSIONS; d++) {
        vertices[num_vertices].coords[d] = 0.0f;
    }
    int center_idx = num_vertices++;
    
    /* 生成每个星臂 */
    for (int arm = 0; arm < arms && num_vertices < MAX_VERTICES; arm++) {
        float angle1 = (float)arm / arms * 2.0f * M_PI;
        float angle2 = (float)arm / arms * 1.5f * M_PI;
        float angle3 = (float)arm / arms * M_PI;
        
        for (int p = 0; p < points_per_arm && num_vertices < MAX_VERTICES; p++) {
            float t = (float)(p + 1) / points_per_arm;
            float r = size * t;
            
            vertices[num_vertices].coords[0] = r * cosf(angle1);
            vertices[num_vertices].coords[1] = r * sinf(angle1);
            vertices[num_vertices].coords[2] = r * cosf(angle2) * 0.5f;
            vertices[num_vertices].coords[3] = r * sinf(angle2) * 0.5f;
            vertices[num_vertices].coords[4] = r * sinf(angle3) * 0.3f;
            
            /* 连接到中心 */
            if (p == 0) {
                vertex_pairs[num_pairs][0] = center_idx;
                vertex_pairs[num_pairs][1] = num_vertices;
                num_pairs++;
            } else {
                /* 连接到前一个点 */
                vertex_pairs[num_pairs][0] = num_vertices - 1;
                vertex_pairs[num_pairs][1] = num_vertices;
                num_pairs++;
            }
            
            num_vertices++;
        }
    }
    
    /* 连接星臂末端 */
    for (int arm = 0; arm < arms; arm++) {
        int idx1 = center_idx + arm * points_per_arm + (points_per_arm - 1);
        int idx2 = center_idx + ((arm + 1) % arms) * points_per_arm + (points_per_arm - 1);
        
        if (idx1 < num_vertices && idx2 < num_vertices && num_pairs < MAX_EDGES) {
            vertex_pairs[num_pairs][0] = idx1;
            vertex_pairs[num_pairs][1] = idx2;
            num_pairs++;
        }
    }
}

/* 生成螺旋形超维结构 */
static void create_hyperspiral(int turns, int points_per_turn, float size) {
    num_vertices = 0;
    num_pairs = 0;
    
    int total_points = turns * points_per_turn;
    if (total_points > MAX_VERTICES) total_points = MAX_VERTICES;
    
    /* 生成螺旋顶点 */
    for (int i = 0; i < total_points && num_vertices < MAX_VERTICES; i++) {
        float t = (float)i / total_points;
        float angle = t * turns * 2.0f * M_PI;
        float height = (t - 0.5f) * size * 1.5f;
        float radius = size * (0.3f + t * 0.7f);
        
        vertices[num_vertices].coords[0] = radius * cosf(angle);
        vertices[num_vertices].coords[1] = radius * sinf(angle);
        vertices[num_vertices].coords[2] = height;
        
        /* 额外的维度做些变形 */
        vertices[num_vertices].coords[3] = radius * cosf(angle * 2.0f) * 0.5f;
        vertices[num_vertices].coords[4] = radius * sinf(angle * 2.0f) * 0.5f;
        
        if (num_vertices > 0 && num_pairs < MAX_EDGES) {
            /* 连接到前一个点 */
            vertex_pairs[num_pairs][0] = num_vertices - 1;
            vertex_pairs[num_pairs][1] = num_vertices;
            num_pairs++;
        }
        
        num_vertices++;
    }
    
    /* 添加螺旋的交叉连接 */
    int points_per_layer = points_per_turn / 2;
    if (points_per_layer > 0) {
        for (int layer = 0; layer < turns*2-1 && num_pairs < MAX_EDGES; layer++) {
            for (int i = 0; i < points_per_layer && num_pairs < MAX_EDGES; i++) {
                int idx1 = layer * points_per_layer + i;
                int idx2 = (layer+1) * points_per_layer + i;
                
                if (idx1 < num_vertices && idx2 < num_vertices) {
                    vertex_pairs[num_pairs][0] = idx1;
                    vertex_pairs[num_pairs][1] = idx2;
                    num_pairs++;
                }
            }
        }
    }
}

/* 创建当前选择的几何体 */
static void create_geometry(int geom_type, float size) {
    switch (geom_type) {
        case GEOM_HYPERCUBE:
            create_hypercube(4, size);
            break;
        case GEOM_HYPERSPHERE:
            create_hypersphere(12, size);
            break;
        case GEOM_HYPERSTAR:
            create_hyperstar(8, 8, size);
            break;
        case GEOM_HYPERSPIRAL:
            create_hyperspiral(12, 10, size);
            break;
        default:
            create_hypercube(4, size);
    }
}

/* HSV到RGB颜色转换，根据hue角度生成彩虹色 */
static uint8_t hsv_to_color(float h, float s, float v) {
    float c = v * s;
    float x = c * (1 - fabsf(fmodf(h / 60.0f, 2) - 1));
    float m = v - c;
    
    float r, g, b;
    if (h < 60) { r = c; g = x; b = 0; }
    else if (h < 120) { r = x; g = c; b = 0; }
    else if (h < 180) { r = 0; g = c; b = x; }
    else if (h < 240) { r = 0; g = x; b = c; }
    else if (h < 300) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }
    
    uint8_t ri = (uint8_t)((r + m) * 255.0f) & 0b11000000;
    uint8_t gi = (uint8_t)((g + m) * 255.0f) & 0b00110000;
    uint8_t bi = (uint8_t)((b + m) * 255.0f) & 0b00001100;
    
    return ri | gi | bi;
}

/* 绘制发光球体 */
static void draw_glow_sphere(int x, int y, int z, float radius, uint8_t core_color, float intensity) {
    pixel_t* volume = voxel_buffer_get(VOXEL_BUFFER_FRONT);
    
    /* 根据亮度衰减计算半径 */
    int max_radius = (int)(radius + 2.0f);
    int inner_radius_sq = (int)(radius * radius / 2.0f);
    int outer_radius_sq = max_radius * max_radius;
    
    int xi = (x - max_radius < 0) ? -x : -max_radius;
    int xs = (x + max_radius >= VOXELS_X) ? (VOXELS_X-1 - x) : max_radius;
    int yi = (y - max_radius < 0) ? -y : -max_radius;
    int ys = (y + max_radius >= VOXELS_Y) ? (VOXELS_Y-1 - y) : max_radius;
    int zi = (z - max_radius < 0) ? -z : -max_radius;
    int zs = (z + max_radius >= VOXELS_Z) ? (VOXELS_Z-1 - z) : max_radius;
    
    /* 根据距离中心的远近应用不同强度的颜色 */
    for (int xx = xi; xx <= xs; ++xx) {
        for (int yy = yi; yy <= ys; ++yy) {
            for (int zz = zi; zz <= zs; ++zz) {
                int dist_sq = xx*xx + yy*yy + zz*zz;
                if (dist_sq <= outer_radius_sq) {
                    if (dist_sq <= inner_radius_sq) {
                        /* 核心部分使用全亮度 */
                        volume[VOXEL_INDEX(x+xx, y+yy, z+zz)] = core_color;
                    } else {
                        /* 外部光晕衰减 */
                        float glow = intensity * (1.0f - (float)dist_sq / outer_radius_sq);
                        uint8_t dimmed = core_color & (uint8_t)(glow * 255);
                        /* 只有当新颜色更亮时才更新像素 */
                        uint8_t current = volume[VOXEL_INDEX(x+xx, y+yy, z+zz)];
                        if ((dimmed & 0xFC) > (current & 0xFC)) {
                            volume[VOXEL_INDEX(x+xx, y+yy, z+zz)] = dimmed;
                        }
                    }
                }
            }
        }
    }
}

/* 初始化粒子系统 */
static void init_particles() {
    for (int i = 0; i < MAX_PARTICLES; i++) {
        particles[i].active = 0;
    }
}

/* 在特定位置创建新粒子 */
static void spawn_particle(float x, float y, float z, float vx, float vy, float vz, float size, float hue, uint8_t type) {
    for (int i = 0; i < MAX_PARTICLES; i++) {
        if (!particles[i].active) {
            particles[i].position[0] = x;
            particles[i].position[1] = y;
            particles[i].position[2] = z;
            particles[i].velocity[0] = vx;
            particles[i].velocity[1] = vy;
            particles[i].velocity[2] = vz;
            particles[i].size = size;
            particles[i].life = 1.0f;
            particles[i].hue = hue;
            particles[i].active = 1;
            particles[i].type = type;
            break;
        }
    }
}

/* 根据粒子类型应用不同的运动模式 */
static void apply_particle_behavior(particle_t *p, float dt, float attraction_points[][3], int num_points, float attraction_strength) {
    switch(p->type) {
        case 0: /* 标准粒子 - 受到吸引力影响 */
            for (int i = 0; i < num_points; i++) {
                float dx = attraction_points[i][0] - p->position[0];
                float dy = attraction_points[i][1] - p->position[1];
                float dz = attraction_points[i][2] - p->position[2];
                float dist_sq = dx*dx + dy*dy + dz*dz;
                
                if (dist_sq < 0.1f) dist_sq = 0.1f;
                float force = attraction_strength / dist_sq;
                
                p->velocity[0] += dx * force * dt;
                p->velocity[1] += dy * force * dt;
                p->velocity[2] += dz * force * dt;
            }
            break;
            
        case 1: /* 螺旋粒子 - 绕最近点旋转 */
            {
                /* 找到最近的吸引点 */
                float min_dist = 1000000.0f;
                int closest = -1;
                for (int i = 0; i < num_points; i++) {
                    float dx = attraction_points[i][0] - p->position[0];
                    float dy = attraction_points[i][1] - p->position[1];
                    float dz = attraction_points[i][2] - p->position[2];
                    float dist_sq = dx*dx + dy*dy + dz*dz;
                    
                    if (dist_sq < min_dist) {
                        min_dist = dist_sq;
                        closest = i;
                    }
                }
                
                if (closest >= 0) {
                    /* 计算到最近点的方向向量 */
                    float dx = attraction_points[closest][0] - p->position[0];
                    float dy = attraction_points[closest][1] - p->position[1];
                    float dz = attraction_points[closest][2] - p->position[2];
                    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                    
                    if (dist > 0.1f) {
                        /* 标准化方向向量 */
                        dx /= dist;
                        dy /= dist;
                        dz /= dist;
                        
                        /* 计算垂直于方向向量的两个轴，构建局部坐标系 */
                        float ax, ay, az, bx, by, bz;
                        
                        /* 找第一个垂直轴 */
                        if (fabsf(dx) < fabsf(dy)) {
                            ax = 0; ay = -dz; az = dy;
                        } else {
                            ax = dz; ay = 0; az = -dx;
                        }
                        float len = sqrtf(ax*ax + ay*ay + az*az);
                        ax /= len; ay /= len; az /= len;
                        
                        /* 找第二个垂直轴 (叉积) */
                        bx = dy*az - dz*ay;
                        by = dz*ax - dx*az;
                        bz = dx*ay - dy*ax;
                        
                        /* 以旋转方式添加速度 */
                        float angle = p->life * 10.0f + time_flow * 5.0f;
                        float strength = attraction_strength * 0.2f;
                        float orbit_radius = fmaxf(dist - 2.0f, 1.0f);
                        float speed = strength / orbit_radius;
                        
                        p->velocity[0] = (ax * cosf(angle) + bx * sinf(angle)) * speed;
                        p->velocity[1] = (ay * cosf(angle) + by * sinf(angle)) * speed;
                        p->velocity[2] = (az * cosf(angle) + bz * sinf(angle)) * speed;
                        
                        /* 添加轻微的向中心力以保持轨道 */
                        p->velocity[0] += dx * strength * 0.05f;
                        p->velocity[1] += dy * strength * 0.05f;
                        p->velocity[2] += dz * strength * 0.05f;
                    }
                }
            }
            break;
            
        case 2: /* 闪烁粒子 - 随机运动 */
            if (rand() % 10 == 0) {
                p->velocity[0] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
                p->velocity[1] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
                p->velocity[2] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
            }
            break;
    }
}

/* 根据时间流逝和吸引力更新粒子 */
static void update_particles(float dt, float attraction_points[][3], int num_points, float attraction_strength) {
    for (int i = 0; i < MAX_PARTICLES; i++) {
        if (particles[i].active) {
            /* 更新寿命 */
            particles[i].life -= dt * 0.2f * (1.0f + 0.5f * ((float)rand() / RAND_MAX));
            if (particles[i].life <= 0) {
                particles[i].active = 0;
                continue;
            }
            
            /* 应用粒子特定行为 */
            apply_particle_behavior(&particles[i], dt, attraction_points, num_points, attraction_strength);
            
            /* 应用阻尼 */
            float damping = 0.98f;
            particles[i].velocity[0] *= damping;
            particles[i].velocity[1] *= damping;
            particles[i].velocity[2] *= damping;
            
            /* 更新位置 */
            particles[i].position[0] += particles[i].velocity[0] * dt;
            particles[i].position[1] += particles[i].velocity[1] * dt;
            particles[i].position[2] += particles[i].velocity[2] * dt;
            
            /* 边界检查 */
            if (particles[i].position[0] < 0 || particles[i].position[0] >= VOXELS_X ||
                particles[i].position[1] < 0 || particles[i].position[1] >= VOXELS_Y ||
                particles[i].position[2] < 0 || particles[i].position[2] >= VOXELS_Z) {
                particles[i].active = 0;
            } else {
                /* 绘制粒子 */
                uint8_t color;
                float glow_intensity;
                
                /* 根据类型设置不同的视觉效果 */
                switch(particles[i].type) {
                    case 1: /* 明亮的轨道粒子 */
                        color = hsv_to_color(
                            particles[i].hue + sinf(time_flow * 2.0f) * 20.0f, 
                            1.0f, 
                            fmaxf(0.6f, particles[i].life)
                        );
                        glow_intensity = particles[i].life * 1.2f;
                        break;
                        
                    case 2: /* 闪烁粒子 */
                        {
                            float flicker = 0.5f + 0.5f * sinf(time_flow * 20.0f + i * 1.3f);
                            color = hsv_to_color(
                                particles[i].hue, 
                                0.8f, 
                                particles[i].life * flicker
                            );
                            glow_intensity = particles[i].life * flicker;
                        }
                        break;
                        
                    default: /* 标准粒子 */
                        color = hsv_to_color(
                            particles[i].hue, 
                            1.0f, 
                            particles[i].life
                        );
                        glow_intensity = particles[i].life;
                }
                
                draw_glow_sphere(
                    (int)particles[i].position[0],
                    (int)particles[i].position[1],
                    (int)particles[i].position[2],
                    particles[i].size * particles[i].life * 1.5f,  /* 放大粒子大小 */
                    color,
                    glow_intensity
                );
            }
        }
    }
}

/* 从超维度投影到3D */
static void project_vertices(vec3_t projected[], mfloat_t matrix[]) {
    float dist = 10.0f;  /* 增加投影距离，扩大视觉范围 */
    float volume_center[3] = {VOXELS_X/2, VOXELS_Y/2, VOXELS_Z/2};
    
    for (int i = 0; i < num_vertices; i++) {
        /* 选取前4个维度进行4D->3D投影 */
        float temp[4];
        for (int d = 0; d < 4; d++) {
            temp[d] = vertices[i].coords[d];
        }
        
        /* 应用变换矩阵 */
        float transformed[4];
        vec3_transform(transformed, temp, matrix);
        
        /* 4D->3D透视投影 */
        float w = transformed[3] + dist;
        float scale = 22.0f / (w + 0.1f);  /* 增大投影系数，扩大可视物体 */
        
        projected[i].x = transformed[0] * scale + volume_center[0];
        projected[i].y = transformed[1] * scale + volume_center[1];
        projected[i].z = transformed[2] * scale + volume_center[2];
        
        /* 第5维度用于脉动大小 */
        if (MAX_DIMENSIONS > 4) {
            projected[i].x += vertices[i].coords[4] * sinf(time_flow * 3.0f) * 0.4f;
            projected[i].y += vertices[i].coords[4] * cosf(time_flow * 2.0f) * 0.4f;
            projected[i].z += vertices[i].coords[4] * sinf(time_flow * 2.5f) * 0.4f;
        }
    }
}

int main(int argc, char** argv) {
    srand(time(NULL));
    
    if (!voxel_buffer_map()) {
        exit(1);
    }
    
    input_set_nonblocking();
    
    /* 创建初始几何体 */
    create_geometry(current_geometry, 1.5f);  /* 增大几何体基础尺寸 */
    
    /* 初始化粒子系统 */
    init_particles();
    
    /* 主循环变量 */
    float rotation_speeds[6] = {0.011f, 0.017f, 0.013f, 0.007f, 0.005f, 0.003f};
    float rotations[6] = {0};
    mfloat_t matrix[MAT4_SIZE];
    vec3_t projected_vertices[MAX_VERTICES];
    float attraction_points[MAX_VERTICES][3];
    int num_attraction_points = 0;
    
    int vertex_mode = 1;    /* 0: 无顶点, 1: 有限顶点, 2: 所有顶点 */
    int edge_mode = 1;      /* 0: 无边, 1: 实线, 2: 粒子流 */
    int particle_mode = 1;  /* 0: 关, 1: 环绕, 2: 爆发 */
    float particle_intensity = 1.0f;
    float geometry_scale = 1.5f;  /* 初始几何体尺寸 */
    
    /* 主循环 */
    for (int ch = 0; ch != 27; ch = getchar()) {
        pixel_t* volume = voxel_buffer_get(VOXEL_BUFFER_FRONT);
        voxel_buffer_clear(volume);
        
        /* 响应键盘控制 */
        if (ch == 'v') vertex_mode = (vertex_mode + 1) % 3;
        if (ch == 'e') edge_mode = (edge_mode + 1) % 3;
        if (ch == 'p') particle_mode = (particle_mode + 1) % 3;
        if (ch == '+' && particle_intensity < 2.0f) particle_intensity += 0.1f;
        if (ch == '-' && particle_intensity > 0.1f) particle_intensity -= 0.1f;
        if (ch == 'g') {
            current_geometry = (current_geometry + 1) % NUM_GEOM_TYPES;
            create_geometry(current_geometry, geometry_scale);
        }
        if (ch == '>' && geometry_scale < 3.0f) {
            geometry_scale += 0.2f;
            create_geometry(current_geometry, geometry_scale);
        }
        if (ch == '<' && geometry_scale > 0.5f) {
            geometry_scale -= 0.2f;
            create_geometry(current_geometry, geometry_scale);
        }
        
        /* 更新时间流 */
        time_flow += TIME_SCALE;
        
        /* 更新旋转矩阵 */
        for (int i = 0; i < 6; i++) {
            rotations[i] = fmodf(rotations[i] + rotation_speeds[i], 2 * M_PI);
        }
        
        /* 创建复杂的旋转矩阵 - 在多个平面上旋转 */
        mat4_identity(matrix);
        
        /* XY平面旋转 */
        mat4_rotation_z(matrix, rotations[0]);
        
        /* XZ平面旋转 */
        mfloat_t temp[MAT4_SIZE];
        mat4_rotation_y(temp, rotations[1]);
        mat4_multiply(matrix, matrix, temp);
        
        /* YZ平面旋转 */
        mat4_rotation_x(temp, rotations[2]);
        mat4_multiply(matrix, matrix, temp);
        
        /* 投影顶点到3D空间 */
        project_vertices(projected_vertices, matrix);
        
        /* 保存吸引点 */
        num_attraction_points = 0;
        if (particle_mode > 0) {
            for (int i = 0; i < num_vertices && num_attraction_points < MAX_VERTICES; i++) {
                if (i % ((vertex_mode == 0) ? 4 : 1) == 0) {
                    attraction_points[num_attraction_points][0] = projected_vertices[i].x;
                    attraction_points[num_attraction_points][1] = projected_vertices[i].y;
                    attraction_points[num_attraction_points][2] = projected_vertices[i].z;
                    num_attraction_points++;
                }
            }
        }
        
        /* 绘制边 */
        if (edge_mode > 0) {
            for (int i = 0; i < num_pairs; i++) {
                int v1 = vertex_pairs[i][0];
                int v2 = vertex_pairs[i][1];
                
                /* 计算边的颜色 - 使用彩虹渐变 */
                float hue = fmodf(time_flow * 20.0f + i * 10.0f, 360.0f);
                uint8_t color = hsv_to_color(hue, 1.0f, 1.0f);
                
                /* 绘制边 */
                if (edge_mode == 1) {
                    /* 实线边 */
                    graphics_draw_line(volume, 
                                      &projected_vertices[v1].x,
                                      &projected_vertices[v2].x, 
                                      color);
                } else {
                    /* 粒子流边 */
                    float dx = projected_vertices[v2].x - projected_vertices[v1].x;
                    float dy = projected_vertices[v2].y - projected_vertices[v1].y;
                    float dz = projected_vertices[v2].z - projected_vertices[v1].z;
                    float len = sqrtf(dx*dx + dy*dy + dz*dz);
                    
                    if (len > 0) {
                        dx /= len;
                        dy /= len;
                        dz /= len;
                        
                        /* 沿线生成粒子 */
                        float t = fmodf(time_flow * 8.0f, 1.0f);
                        float x = projected_vertices[v1].x + dx * len * t;
                        float y = projected_vertices[v1].y + dy * len * t;
                        float z = projected_vertices[v1].z + dz * len * t;
                        
                        /* 随机扰动速度 */
                        float vx = dx * 0.5f + ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
                        float vy = dy * 0.5f + ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
                        float vz = dz * 0.5f + ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
                        
                        /* 生成粒子 */
                        if (rand() % 5 == 0) {
                            spawn_particle(x, y, z, vx, vy, vz, 1.5f, hue, 0);  /* 标准粒子 */
                        }
                    }
                }
            }
        }
        
        /* 绘制顶点 */
        if (vertex_mode > 0) {
            for (int i = 0; i < num_vertices; i++) {
                if (vertex_mode == 1 && i % 4 != 0) continue;
                
                /* 顶点呼吸效果 */
                float breath = 1.0f + 0.3f * sinf(time_flow * 4.0f + i * 0.2f);
                float size = 2.0f * breath;  /* 增大顶点尺寸 */
                
                /* 顶点颜色基于其位置和时间 */
                float hue = fmodf(time_flow * 30.0f + i * 20.0f, 360.0f);
                uint8_t color = hsv_to_color(hue, 1.0f, 1.0f);
                
                /* 绘制脉动球体 */
                draw_glow_sphere(
                    (int)projected_vertices[i].x,
                    (int)projected_vertices[i].y,
                    (int)projected_vertices[i].z,
                    size, color, 1.0f
                );
                
                /* 爆发模式：不定期从顶点喷发粒子 */
                if (particle_mode == 2 && rand() % 30 == 0) {  /* 增加爆发频率 */
                    for (int p = 0; p < 20; p++) {  /* 每次爆发更多粒子 */
                        float angle1 = (float)rand() / RAND_MAX * 2 * M_PI;
                        float angle2 = (float)rand() / RAND_MAX * M_PI;
                        float speed = ((float)rand() / RAND_MAX * 0.5f + 0.5f);
                        
                        float vx = cosf(angle1) * sinf(angle2) * speed;
                        float vy = sinf(angle1) * sinf(angle2) * speed;
                        float vz = cosf(angle2) * speed;
                        
                        uint8_t type = rand() % 3;  /* 随机粒子类型 */
                        
                        spawn_particle(
                            projected_vertices[i].x,
                            projected_vertices[i].y,
                            projected_vertices[i].z,
                            vx, vy, vz,
                            (float)rand() / RAND_MAX + 0.7f,  /* 更大的粒子 */
                            hue + ((float)rand() / RAND_MAX - 0.5f) * 40.0f,
                            type
                        );
                    }
                }
            }
        }
        
        /* 环绕模式：在吸引点周围生成新粒子 */
        if (particle_mode == 1) {
            int spawn_rate = (rand() % 3) + 1;  /* 动态调整生成率 */
            for (int s = 0; s < spawn_rate; s++) {
                int point_idx = rand() % num_attraction_points;
                float radius = 8.0f + sinf(time_flow) * 3.0f;  /* 增加环绕半径 */
                float angle1 = (float)rand() / RAND_MAX * 2 * M_PI;
                float angle2 = (float)rand() / RAND_MAX * M_PI;
                
                float x = attraction_points[point_idx][0] + cosf(angle1) * sinf(angle2) * radius;
                float y = attraction_points[point_idx][1] + sinf(angle1) * sinf(angle2) * radius;
                float z = attraction_points[point_idx][2] + cosf(angle2) * radius;
                
                float hue = fmodf(time_flow * 50.0f, 360.0f);
                
                /* 随机选择粒子类型 */
                uint8_t type = (rand() % 100) < 70 ? 0 : (rand() % 100 < 70 ? 1 : 2);
                
                spawn_particle(
                    x, y, z,
                    ((float)rand() / RAND_MAX - 0.5f) * 0.2f,
                    ((float)rand() / RAND_MAX - 0.5f) * 0.2f,
                    ((float)rand() / RAND_MAX - 0.5f) * 0.2f,
                    (float)rand() / RAND_MAX * 0.5f + 0.8f,  /* 更大的粒子 */
                    hue + ((float)rand() / RAND_MAX - 0.5f) * 20.0f,
                    type
                );
            }
        }
        
        /* 更新粒子系统 */
        float attraction = (particle_mode == 1) ? particle_intensity * 0.02f : -0.01f;
        update_particles(0.12f, attraction_points, num_attraction_points, attraction);
        
        /* 显示当前几何体类型和控制信息 */
        /* 这里只用注释表示，因为在voxel缓冲区中无法显示文本信息 */
        /*
        const char* geom_names[] = {"Hypercube", "Hypersphere", "Hyperstar", "Hyperspiral"};
        printf("Geometry: %s | Scale: %.1f | V: %d | E: %d | P: %d\n", 
               geom_names[current_geometry], geometry_scale, vertex_mode, edge_mode, particle_mode);
        */
        
        voxel_buffer_swap();
        usleep(20000); /* ~50 fps */
    }
    
    voxel_buffer_unmap();
    return 0;
}