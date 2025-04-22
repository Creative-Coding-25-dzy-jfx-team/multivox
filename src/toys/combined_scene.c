#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#include "rammel.h"
#include "input.h"
#include "voxel.h"
#include "mathc.h"
#include "graphics.h"

// 大幅增加点的数量
#define MAX_PARTICLES 1200
#define MAX_BRAIN_NODES 800
#define MAX_NEURON_CONNECTIONS 2400
#define M_PI 3.14159265358979323846
// 脑组织颜色常量 - 基于真实大脑的色调
#define BRAIN_GRAY_MATTER HEXPIX(CC9A88)  // 灰质呈灰粉色
#define BRAIN_WHITE_MATTER HEXPIX(FFDDCC) // 白质呈淡粉色
#define BRAIN_CORTEX HEXPIX(DD8877)       // 皮层层更深的粉红色
#define BRAIN_CEREBELLUM HEXPIX(BB7766)   // 小脑稍深色
#define BRAIN_BRAINSTEM HEXPIX(AA6655)    // 脑干更深色
#define BRAIN_VESSEL HEXPIX(CC3322)       // 血管偏红色

typedef struct
{
    vec3_t position;
    vec3_t velocity;
    float size;
    float life;
    uint8_t colour;
    bool active;
} particle_t;

typedef struct
{
    vec3_t position;
    float size;
    uint8_t colour;
    int connections;
    int connection_ids[10];       // 每个节点最多10个连接
    uint8_t connection_types[10]; // 连接类型
} neuron_t;

// 全局变量
static particle_t particles[MAX_PARTICLES];
static neuron_t neurons[MAX_BRAIN_NODES];
static int active_neurons = 0;
static int connection_map[MAX_NEURON_CONNECTIONS][2]; // 存储连接的两端
static int active_connections = 0;

// 绘制小球体 - 减小点的大小
void draw_small_sphere(int x, int y, int z, float radius, uint8_t colour)
{
    pixel_t *volume = voxel_buffer_get(VOXEL_BUFFER_FRONT);

    int r = (int)ceil(radius);
    int xi = (x - r < 0) ? -x : -r;
    int xs = (x + r >= VOXELS_X) ? (VOXELS_X - 1 - x) : r;
    int yi = (y - r < 0) ? -y : -r;
    int ys = (y + r >= VOXELS_Y) ? (VOXELS_Y - 1 - y) : r;
    int zi = (z - r < 0) ? -z : -r;
    int zs = (z + r >= VOXELS_Z) ? (VOXELS_Z - 1 - z) : r;

    float rosq = radius * radius;

    for (int xx = xi; xx <= xs; ++xx)
    {
        for (int yy = yi; yy <= ys; ++yy)
        {
            for (int zz = zi; zz <= zs; ++zz)
            {
                float lsq = xx * xx + yy * yy + zz * zz;
                if (lsq <= rosq)
                {
                    volume[VOXEL_INDEX(x + xx, y + yy, z + zz)] = colour;
                }
            }
        }
    }
}

// 绘制点而不是球体 - 用于提高性能
void draw_point(int x, int y, int z, uint8_t colour)
{
    if (x >= 0 && x < VOXELS_X && y >= 0 && y < VOXELS_Y && z >= 0 && z < VOXELS_Z)
    {
        pixel_t *volume = voxel_buffer_get(VOXEL_BUFFER_FRONT);
        volume[VOXEL_INDEX(x, y, z)] = colour;
    }
}

// 绘制发光点 - 更高效的微小发光效果
void draw_glow_point(int x, int y, int z, uint8_t colour, float intensity)
{
    pixel_t *volume = voxel_buffer_get(VOXEL_BUFFER_FRONT);

    // 中心点
    draw_point(x, y, z, colour);

    // 周围点用较暗颜色
    uint8_t dim_colour = (colour & 0xFC) >> 1; // 减半亮度

    // 六个相邻点
    if (x > 0)
        draw_point(x - 1, y, z, dim_colour);
    if (x < VOXELS_X - 1)
        draw_point(x + 1, y, z, dim_colour);
    if (y > 0)
        draw_point(x, y - 1, z, dim_colour);
    if (y < VOXELS_Y - 1)
        draw_point(x, y + 1, z, dim_colour);
    if (z > 0)
        draw_point(x, y, z - 1, dim_colour);
    if (z < VOXELS_Z - 1)
        draw_point(x, y, z + 1, dim_colour);
}
// 根据脑区域获取适当的颜色
uint8_t get_brain_color(int region, float variation)
{
    // 添加轻微随机变化使色彩更自然
    uint8_t color;

    switch (region)
    {
    case 0: // 前额叶区域
        color = BRAIN_CORTEX;
        break;
    case 1: // 顶叶和颞叶区域
        color = BRAIN_GRAY_MATTER;
        break;
    case 2: // 枕叶区域
        color = BRAIN_CORTEX;
        break;
    case 3: // 小脑区域
        color = BRAIN_CEREBELLUM;
        break;
    case 4: // 脑干区域
        color = BRAIN_BRAINSTEM;
        break;
    default:
        color = BRAIN_GRAY_MATTER;
    }

    // 应用微小的颜色变化以增加自然感
    if (variation > 0.5f)
    {
        // 更亮一点
        return (color & 0xFC) | 0x01;
    }
    else if (variation < -0.5f)
    {
        // 更暗一点
        return (color & 0xFC) >> 1;
    }

    return color;
}
uint8_t get_vessel_color(bool is_long_connection, float pulse)
{
    if (is_long_connection)
    {
        // 长距离连接使用深红色（主要静脉）
        return BRAIN_VESSEL;
    }
    else
    {
        // 短距离连接使用较浅红色（毛细血管）
        return (BRAIN_VESSEL & 0xFC) | 0x01;
    }
}

// 脉冲粒子颜色
uint8_t get_pulse_color(float intensity)
{
    // 神经脉冲使用偏白色，表示电信号
    uint8_t white = HEXPIX(FFFFFF);
    uint8_t red = BRAIN_VESSEL;

    // 根据强度混合白色和红色
    if (intensity > 0.8f)
    {
        return white;
    }
    else if (intensity > 0.5f)
    {
        return (white & 0xF0) | (red & 0x0F);
    }
    else
    {
        return red;
    }
}

// HSV到RGB颜色转换
uint8_t hsv_to_color(float h, float s, float v)
{
    h = fmodf(h, 360.0f);
    s = fminf(fmaxf(s, 0.0f), 1.0f);
    v = fminf(fmaxf(v, 0.0f), 1.0f);

    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;

    float r, g, b;
    if (h < 60.0f)
    {
        r = c;
        g = x;
        b = 0;
    }
    else if (h < 120.0f)
    {
        r = x;
        g = c;
        b = 0;
    }
    else if (h < 180.0f)
    {
        r = 0;
        g = c;
        b = x;
    }
    else if (h < 240.0f)
    {
        r = 0;
        g = x;
        b = c;
    }
    else if (h < 300.0f)
    {
        r = x;
        g = 0;
        b = c;
    }
    else
    {
        r = c;
        g = 0;
        b = x;
    }

    uint8_t ri = (uint8_t)((r + m) * 255.0f) & 0b11000000;
    uint8_t gi = (uint8_t)((g + m) * 255.0f) & 0b00110000;
    uint8_t bi = (uint8_t)((b + m) * 255.0f) & 0b00001100;

    return ri | gi | bi;
}

// 初始化粒子系统
void init_particles()
{
    for (int i = 0; i < MAX_PARTICLES; i++)
    {
        particles[i].active = false;
    }
}

// 创建新粒子
void spawn_particle(float x, float y, float z, float vx, float vy, float vz, float size, float hue)
{
    for (int i = 0; i < MAX_PARTICLES; i++)
    {
        if (!particles[i].active)
        {
            particles[i].position.x = x;
            particles[i].position.y = y;
            particles[i].position.z = z;
            particles[i].velocity.x = vx;
            particles[i].velocity.y = vy;
            particles[i].velocity.z = vz;
            particles[i].size = size;
            particles[i].life = 1.0f;
            particles[i].colour = hsv_to_color(hue, 1.0f, 1.0f);
            particles[i].active = true;
            return;
        }
    }
}

// 基于神经网络布局算法生成脑结构
void create_brain_structure()
{
    active_neurons = 0;
    active_connections = 0;

    // 清空神经元和连接
    memset(neurons, 0, sizeof(neurons));
    memset(connection_map, 0, sizeof(connection_map));

    // 中心点
    float center_x = VOXELS_X / 2.0f;
    float center_y = VOXELS_Y / 2.0f;
    float center_z = VOXELS_Z / 2.0f;

    // 创建脑形状的参数
    float brain_width = VOXELS_X * 0.35f;
    float brain_height = VOXELS_Y * 0.28f;
    float brain_depth = VOXELS_Z * 0.33f;

    // 使用参数方程和柏林噪声模拟脑表面皱褶
    for (int i = 0; i < MAX_BRAIN_NODES; i++)
    {
        float theta, phi, r;

        // 针对不同区域创建不同密度的点
        if (i < MAX_BRAIN_NODES * 0.3f)
        {
            // 前额叶区域 - 更多点
            theta = ((float)rand() / RAND_MAX) * M_PI * 0.6f;
            phi = ((float)rand() / RAND_MAX) * M_PI * 0.7f + M_PI * 0.15f;
            r = 0.85f + ((float)rand() / RAND_MAX) * 0.15f;
        }
        else if (i < MAX_BRAIN_NODES * 0.6f)
        {
            // 顶叶和颞叶区域
            theta = ((float)rand() / RAND_MAX) * M_PI * 0.7f + M_PI * 0.15f;
            phi = ((float)rand() / RAND_MAX) * M_PI * 1.2f;
            r = 0.8f + ((float)rand() / RAND_MAX) * 0.2f;
        }
        else if (i < MAX_BRAIN_NODES * 0.8f)
        {
            // 枕叶区域
            theta = ((float)rand() / RAND_MAX) * M_PI * 0.5f + M_PI * 0.5f;
            phi = ((float)rand() / RAND_MAX) * M_PI + M_PI * 0.5f;
            r = 0.88f + ((float)rand() / RAND_MAX) * 0.12f;
        }
        else
        {
            // 小脑区域
            theta = ((float)rand() / RAND_MAX) * M_PI * 0.3f + M_PI * 0.6f;
            phi = ((float)rand() / RAND_MAX) * M_PI * 0.6f + M_PI * 0.7f;
            r = 0.75f + ((float)rand() / RAND_MAX) * 0.15f;
        }

        // 椭球参数方程
        float x = r * brain_width * sinf(theta) * cosf(phi);
        float y = r * brain_height * sinf(theta) * sinf(phi);
        float z = r * brain_depth * cosf(theta);

        // 添加脑沟和脑回的不规则结构 (使用多个正弦函数模拟褶皱)
        float wrinkle_amplitude = brain_width * 0.06f;
        float wrinkle_freq = 8.0f;

        float noise_x = wrinkle_amplitude * sinf(theta * wrinkle_freq) * cosf(phi * wrinkle_freq);
        float noise_y = wrinkle_amplitude * sinf(phi * wrinkle_freq) * sinf(theta * wrinkle_freq);
        float noise_z = wrinkle_amplitude * cosf((theta + phi) * wrinkle_freq * 0.5f);

        // 中心脑沟
        if (fabsf(phi - M_PI) < 0.4f)
        {
            // 中央纵裂
            y += (phi - M_PI) * brain_height * 0.1f;
        }

        // 存储神经元
        neurons[active_neurons].position.x = center_x + x + noise_x;
        neurons[active_neurons].position.y = center_y + y + noise_y;
        neurons[active_neurons].position.z = center_z + z + noise_z;

        // 随机大小 - 比原来小
        neurons[active_neurons].size = 0.5f + ((float)rand() / RAND_MAX) * 0.8f;

        // 根据脑区域分配颜色
        // 替换之前的"根据脑区域分配颜色"部分
        float variation = ((float)rand() / RAND_MAX - 0.5f);
        int region_type;

        if (i < MAX_BRAIN_NODES * 0.3f)
        {
            // 前额叶
            region_type = 0;
        }
        else if (i < MAX_BRAIN_NODES * 0.6f)
        {
            // 顶叶和颞叶
            region_type = 1;
        }
        else if (i < MAX_BRAIN_NODES * 0.8f)
        {
            // 枕叶
            region_type = 2;
        }
        else if (i < MAX_BRAIN_NODES * 0.95f)
        {
            // 小脑
            region_type = 3;
        }
        else
        {
            // 脑干
            region_type = 4;
        }

        neurons[active_neurons].colour = get_brain_color(region_type, variation);
        neurons[active_neurons].connections = 0;

        active_neurons++;
    }

    // 创建连接 - 使用小世界网络模型
    // 1. 近邻连接
    for (int i = 0; i < active_neurons; i++)
    {
        // 每个神经元与4-8个邻近神经元相连
        int local_connections = 3 + (rand() % 5);

        // 寻找最近的神经元
        typedef struct
        {
            int index;
            float distance;
        } neighbor_t;

        neighbor_t neighbors[MAX_BRAIN_NODES];

        // 计算与所有其他神经元的距离
        for (int j = 0; j < active_neurons; j++)
        {
            if (j == i)
                continue;

            float dx = neurons[i].position.x - neurons[j].position.x;
            float dy = neurons[i].position.y - neurons[j].position.y;
            float dz = neurons[i].position.z - neurons[j].position.z;

            neighbors[j].index = j;
            neighbors[j].distance = dx * dx + dy * dy + dz * dz;
        }

        // 简单的冒泡排序找出最近的几个
        for (int a = 0; a < active_neurons - 1; a++)
        {
            for (int b = 0; b < active_neurons - a - 1; b++)
            {
                if (neighbors[b].distance > neighbors[b + 1].distance)
                {
                    neighbor_t temp = neighbors[b];
                    neighbors[b] = neighbors[b + 1];
                    neighbors[b + 1] = temp;
                }
            }
        }

        // 连接到最近的几个神经元
        for (int c = 0; c < local_connections && c < active_neurons - 1; c++)
        {
            int target = neighbors[c].index;

            // 避免重复连接
            bool already_connected = false;
            for (int k = 0; k < neurons[i].connections; k++)
            {
                if (neurons[i].connection_ids[k] == target)
                {
                    already_connected = true;
                    break;
                }
            }

            if (!already_connected && active_connections < MAX_NEURON_CONNECTIONS)
            {
                // 添加连接信息到神经元
                neurons[i].connection_ids[neurons[i].connections] = target;
                neurons[i].connection_types[neurons[i].connections] = 1; // 1=近距离连接
                neurons[i].connections++;

                // 添加到连接地图
                connection_map[active_connections][0] = i;
                connection_map[active_connections][1] = target;
                active_connections++;
            }
        }
    }

    // 2. 长距离连接 (小世界特性)
    int long_connections = MAX_NEURON_CONNECTIONS * 0.1f; // 10%是长距离连接
    for (int i = 0; i < long_connections && active_connections < MAX_NEURON_CONNECTIONS; i++)
    {
        int source = rand() % active_neurons;
        int target = rand() % active_neurons;

        // 避免自连接和重复连接
        if (source != target)
        {
            bool already_connected = false;
            for (int j = 0; j < neurons[source].connections; j++)
            {
                if (neurons[source].connection_ids[j] == target)
                {
                    already_connected = true;
                    break;
                }
            }

            if (!already_connected && neurons[source].connections < 10)
            {
                neurons[source].connection_ids[neurons[source].connections] = target;
                neurons[source].connection_types[neurons[source].connections] = 2; // 2=长距离连接
                neurons[source].connections++;

                connection_map[active_connections][0] = source;
                connection_map[active_connections][1] = target;
                active_connections++;
            }
        }
    }

    printf("创建大脑结构: %d 个神经元, %d 个连接\n", active_neurons, active_connections);
}

// 更新粒子系统
void update_particles(float dt)
{
    for (int i = 0; i < MAX_PARTICLES; i++)
    {
        if (particles[i].active)
        {
            // 更新位置
            particles[i].position.x += particles[i].velocity.x * dt;
            particles[i].position.y += particles[i].velocity.y * dt;
            particles[i].position.z += particles[i].velocity.z * dt;

            // 生命周期衰减
            particles[i].life -= dt * 0.2f;

            // 检查是否应该消失
            if (particles[i].life <= 0.0f)
            {
                particles[i].active = false;
                continue;
            }

            // 检查边界
            if (particles[i].position.x < 0 || particles[i].position.x >= VOXELS_X ||
                particles[i].position.y < 0 || particles[i].position.y >= VOXELS_Y ||
                particles[i].position.z < 0 || particles[i].position.z >= VOXELS_Z)
            {
                particles[i].active = false;
                continue;
            }

            // 绘制粒子 - 现在使用更小的点
            float size = particles[i].size * particles[i].life;
            if (size > 1.0f)
            {
                draw_small_sphere(
                    (int)particles[i].position.x,
                    (int)particles[i].position.y,
                    (int)particles[i].position.z,
                    size * 0.7f, // 减小尺寸
                    particles[i].colour);
            }
            else
            {
                // 小尺寸用点绘制以提高性能
                draw_glow_point(
                    (int)particles[i].position.x,
                    (int)particles[i].position.y,
                    (int)particles[i].position.z,
                    particles[i].colour,
                    particles[i].life);
            }
        }
    }
}

// 模拟神经脉冲沿着连接传播
void simulate_neural_pulse(int neuron_idx, float time_flow)
{
    // 在连接上随机生成信号粒子
    for (int c = 0; c < neurons[neuron_idx].connections; c++)
    {
        int target = neurons[neuron_idx].connection_ids[c];

        // 根据连接类型和时间决定是否生成脉冲
        bool generate_pulse = false;
        if (neurons[neuron_idx].connection_types[c] == 1)
        {                                        // 短连接
            generate_pulse = (rand() % 100 < 4); // 4%概率
        }
        else
        {                                        // 长连接
            generate_pulse = (rand() % 100 < 2); // 2%概率
        }

        if (generate_pulse)
        {
            // 起点
            float start_x = neurons[neuron_idx].position.x;
            float start_y = neurons[neuron_idx].position.y;
            float start_z = neurons[neuron_idx].position.z;

            // 方向向量
            float dx = neurons[target].position.x - start_x;
            float dy = neurons[target].position.y - start_y;
            float dz = neurons[target].position.z - start_z;

            // 计算路径总长度
            float path_length = sqrtf(dx * dx + dy * dy + dz * dz);

            // 生成多个粒子沿着连接
            int num_pulses = 1 + (rand() % 3); // 1-3个脉冲
            for (int p = 0; p < num_pulses; p++)
            {
                // 随机位置比例
                float t = (float)rand() / RAND_MAX;

                // 微小随机偏移让路径更自然
                float jitter = 0.8f;
                float offset_x = ((float)rand() / RAND_MAX - 0.5f) * jitter;
                float offset_y = ((float)rand() / RAND_MAX - 0.5f) * jitter;
                float offset_z = ((float)rand() / RAND_MAX - 0.5f) * jitter;

                // 速度 - 沿着连接方向
                float speed = 0.8f + (float)rand() / RAND_MAX * 0.4f;
                float dist = path_length * 0.01f; // 标准化因子
                float vel_x = dx / path_length * speed;
                float vel_y = dy / path_length * speed;
                float vel_z = dz / path_length * speed;

                // 生成粒子
                float hue = fmodf(time_flow * 50.0f, 360.0f);
                if (neurons[neuron_idx].connection_types[c] == 2)
                {
                    // 长距离连接使用不同颜色
                    hue = fmodf(hue + 180.0f, 360.0f);
                }

                spawn_particle(
                    start_x + dx * t + offset_x,
                    start_y + dy * t + offset_y,
                    start_z + dz * t + offset_z,
                    vel_x, vel_y, vel_z,
                    0.7f, // 更小的粒子
                    hue + ((float)rand() / RAND_MAX - 0.5f) * 30.0f);
            }
        }
    }
}

// 主函数
int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    srand(time(NULL));

    if (!voxel_buffer_map())
    {
        exit(1);
    }

    input_set_nonblocking();

    // 初始化
    init_particles();
    create_brain_structure();

    float time_flow = 0.0f;
    float brain_pulse = 0.0f;

    // 主循环
    for (int ch = 0; ch != 27; ch = getchar())
    {
        pixel_t *volume = voxel_buffer_get(VOXEL_BUFFER_FRONT);
        voxel_buffer_clear(volume);

        // 更新时间相关变量
        time_flow += 0.05f;
        brain_pulse = sinf(time_flow) * 0.5f + 0.5f; // 0-1之间脉动

        // 绘制所有神经元节点和连接
        for (int i = 0; i < active_neurons; i++)
        {
            // 绘制神经元 - 现在使用更小的球体
            float node_size = neurons[i].size * (0.6f + brain_pulse * 0.3f);
            uint8_t node_color = neurons[i].colour;

            if (node_size <= 1.0f)
            {
                draw_glow_point(
                    (int)neurons[i].position.x,
                    (int)neurons[i].position.y,
                    (int)neurons[i].position.z,
                    node_color,
                    0.7f + brain_pulse * 0.3f);
            }
            else
            {
                draw_small_sphere(
                    (int)neurons[i].position.x,
                    (int)neurons[i].position.y,
                    (int)neurons[i].position.z,
                    node_size,
                    node_color);
            }
        }

        // 绘制连接
        for (int i = 0; i < active_connections; i++)
        {
            int source = connection_map[i][0];
            int target = connection_map[i][1];

            // 随机变化颜色
            float hue = fmodf(time_flow * 20.0f + i, 360.0f);
            uint8_t line_color;

            if (neurons[source].connection_types[0] == 2)
            { // 长距离连接
                line_color = get_vessel_color(true, brain_pulse);
            }
            else
            { // 短距离连接
                line_color = get_vessel_color(false, brain_pulse);
            }

            // 绘制连接线 - 淡化线条颜色
            graphics_draw_line(
                volume,
                (float *)&neurons[source].position,
                (float *)&neurons[target].position,
                line_color);
        }

        // 模拟神经元激活
        for (int i = 0; i < active_neurons; i++)
        {
            // 每帧随机选择一些神经元产生脉冲
            if (rand() % 100 < 5)
            { // 5%的概率每帧激活
                simulate_neural_pulse(i, time_flow);
            }
        }

        // 随机在某些脑区域爆发更多的活动
        if (rand() % 40 == 0)
        { // 更频繁的爆发
            // 选择一个区域中心点
            int region_center = rand() % active_neurons;
            int num_activated = 3 + rand() % 5; // 3-7个神经元

            // 寻找最近的几个神经元一起激活
            for (int n = 0; n < num_activated; n++)
            {
                // 找到一个邻近神经元
                int target_neuron = (region_center + n) % active_neurons;

                // 产生更强烈的活动
                for (int p = 0; p < 4 + rand() % 3; p++)
                {
                    float angle1 = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
                    float angle2 = ((float)rand() / RAND_MAX) * M_PI;
                    float speed = ((float)rand() / RAND_MAX) * 0.7f + 0.3f;

                    float vx = cosf(angle1) * sinf(angle2) * speed;
                    float vy = sinf(angle1) * sinf(angle2) * speed;
                    float vz = cosf(angle2) * speed;

                    float hue = fmodf(time_flow * 50.0f, 360.0f);

                    // 爆发出一些小粒子
                    spawn_particle(
                        neurons[target_neuron].position.x,
                        neurons[target_neuron].position.y,
                        neurons[target_neuron].position.z,
                        vx * 0.2f, vy * 0.2f, vz * 0.2f,
                        0.5f + (float)rand() / RAND_MAX * 0.5f, // 更小的粒子
                        hue + ((float)rand() / RAND_MAX - 0.5f) * 30.0f);
                }

                // 同时激活这个神经元的所有连接
                simulate_neural_pulse(target_neuron, time_flow);
            }
        }

        // 更新粒子系统
        update_particles(1.0f);

        usleep(30000); // ~33fps
    }

    voxel_buffer_unmap();

    return 0;
}