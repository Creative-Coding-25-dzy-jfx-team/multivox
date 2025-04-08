#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include "array.h"
#include "mathc.h"
#include "rammel.h"
#include "input.h"
#include "graphics.h"
#include "model.h"
#include "voxel.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 与原文件相同的超立方体顶点
static float tess_vertices[16][VEC4_SIZE] = {
    { 1,  1,  1,  1}, { 1,  1,  1, -1}, { 1,  1, -1,  1}, { 1,  1, -1, -1},
    { 1, -1,  1,  1}, { 1, -1,  1, -1}, { 1, -1, -1,  1}, { 1, -1, -1, -1},
    {-1,  1,  1,  1}, {-1,  1,  1, -1}, {-1,  1, -1,  1}, {-1,  1, -1, -1},
    {-1, -1,  1,  1}, {-1, -1,  1, -1}, {-1, -1, -1,  1}, {-1, -1, -1, -1},
};

// 面数据，与原文件一致（用于填充）
#ifdef __GNUC__
__attribute__((unused))
#endif
static int tess_faces[24][4] = {
    { 0,  1,  5,  4}, { 0,  2,  6,  4}, { 0,  8, 12,  4},
    { 0,  2,  3,  1}, { 0,  1,  9,  8}, { 0,  2, 10,  8},
    { 1,  3,  7,  5}, { 1,  9, 13,  5}, { 1,  9, 11,  3},
    { 2,  3,  7,  6}, {11, 10,  2,  3}, { 2, 10, 14,  6},
    { 3, 11, 15,  7}, { 4, 12, 13,  5}, { 4,  6, 14, 12},
    { 4,  6,  7,  5}, { 5,  7, 15, 13}, { 7,  6, 14, 15},
    { 8, 10, 14, 12}, { 8,  9, 13, 12}, { 9,  8, 10, 11},
    { 9, 11, 15, 13}, {10, 11, 15, 14}, {12, 14, 15, 13},
};

// 边数据，与原文件一致
static int tess_edges[32][2] = {
    {0, 1}, {0, 2}, {0, 4}, {0, 8},
    {1, 3}, {1, 5}, {1, 9},
    {2, 3}, {2, 6}, {2, 10},
    {3, 7}, {3, 11},
    {4, 5}, {4, 6}, {4, 12},
    {5, 7}, {5, 13},
    {6, 7}, {6, 14},
    {7, 15},
    {8, 9}, {8, 10}, {8, 12},
    {9, 11}, {9, 13},
    {10, 11}, {10, 14},
    {11, 15},
    {12, 13}, {12, 14},
    {13, 15},
    {14, 15}
};

static const pixel_t colours[] = {
    HEXPIX(FF0000),
    HEXPIX(FFFF00),
    HEXPIX(00FF00),
    HEXPIX(00FFFF),
    HEXPIX(0000FF),
};

// 原文件中的变换函数
static float *vec4_transform(float *result, float *v0, float *m0) {
    float x = v0[0];
    float y = v0[1];
    float z = v0[2];
    float w = v0[3];

    result[0] = m0[0] * x + m0[4] * y + m0[ 8] * z + m0[12] * w;
    result[1] = m0[1] * x + m0[5] * y + m0[ 9] * z + m0[13] * w;
    result[2] = m0[2] * x + m0[6] * y + m0[10] * z + m0[14] * w;
    result[3] = m0[3] * x + m0[7] * y + m0[11] * z + m0[15] * w;

    return result;
}

int main(int argc, char** argv) {
    if (!voxel_buffer_map()) {
        exit(1);
    }

    // 体素缓冲区中心
    float centre[VEC3_SIZE] = {VOXELS_X / 2, VOXELS_Y / 2, VOXELS_Z / 2};
    // 两个超立方体的旋转角度
    float rotation1[VEC3_SIZE] = {0, 0, 0};
    float rotation2[VEC3_SIZE] = {0, 0, 0};

    // 变换后的顶点数据
    vec3_t transformed1[count_of(tess_vertices)];
    vec3_t transformed2[count_of(tess_vertices)];

    mfloat_t matrix1[MAT4_SIZE];
    mfloat_t matrix2[MAT4_SIZE];

    // 视角参数与透视投影常数
    float dist = 4;
    float fovt = dist * 16;
    
    input_set_nonblocking();
    bool show_faces = false;
    int frame = 0;

    while (1) {
        int ch = getchar();
        if (ch == 27) break;  // ESC键退出
        if (ch == 'f') {
            show_faces = !show_faces;
        }

        pixel_t* volume = voxel_buffer_get(VOXEL_BUFFER_BACK);
        voxel_buffer_clear(volume);

        // 更新旋转角度
        rotation1[0] = fmodf(rotation1[0] + 0.013f, 2 * M_PI);
        rotation1[2] = fmodf(rotation1[2] + 0.017f, 2 * M_PI);
        rotation2[0] = fmodf(rotation2[0] + 0.020f, 2 * M_PI);
        rotation2[1] = fmodf(rotation2[1] + 0.015f, 2 * M_PI);

        // 创新点：动态缩放因子，产生“呼吸”效果
        float scale = 1.0f + 0.3f * sinf(frame * 0.1f);

        // 第一个超立方体：对顶点进行缩放处理
        mat4_identity(matrix1);
        mat4_apply_rotation(matrix1, rotation1);
        for (uint i = 0; i < count_of(tess_vertices); ++i) {
            float scaled_vertex[VEC4_SIZE];
            for (int j = 0; j < VEC4_SIZE; j++) {
                scaled_vertex[j] = tess_vertices[i][j] * scale;
            }
            float vp[VEC4_SIZE];
            vec4_transform(vp, scaled_vertex, matrix1);
            float s = fovt / (vp[2] + dist);
            transformed1[i].x =  vp[3] * s + centre[0];
            transformed1[i].y = -vp[1] * s + centre[1];
            transformed1[i].z = -vp[0] * s + centre[2];
        }

        // 第二个超立方体：不使用缩放，另外平移一定偏移
        float offset = 20;
        mat4_identity(matrix2);
        mat4_apply_rotation(matrix2, rotation2);
        for (uint i = 0; i < count_of(tess_vertices); ++i) {
            float vp[VEC4_SIZE];
            vec4_transform(vp, tess_vertices[i], matrix2);
            float s = fovt / (vp[2] + dist);
            transformed2[i].x =  vp[3] * s + centre[0] + offset;
            transformed2[i].y = -vp[1] * s + centre[1];
            transformed2[i].z = -vp[0] * s + centre[2];
        }
        
        // 如启用面渲染则绘制两个超立方体的面
        if (show_faces) {
            for (uint i = 0; i < count_of(tess_faces); ++i) {
                pixel_t col = colours[i % count_of(colours)] & 0b01101101;
                graphics_triangle_colour(col);
                graphics_draw_triangle(volume, &transformed1[tess_faces[i][0]],
                                            &transformed1[tess_faces[i][1]],
                                            &transformed1[tess_faces[i][2]]);
                graphics_draw_triangle(volume, &transformed1[tess_faces[i][0]],
                                            &transformed1[tess_faces[i][2]],
                                            &transformed1[tess_faces[i][3]]);
                // 第二个超立方体面，颜色稍作对比
                graphics_triangle_colour(col ^ 0x00FFFFFF);
                graphics_draw_triangle(volume, &transformed2[tess_faces[i][0]],
                                            &transformed2[tess_faces[i][1]],
                                            &transformed2[tess_faces[i][2]]);
                graphics_draw_triangle(volume, &transformed2[tess_faces[i][0]],
                                            &transformed2[tess_faces[i][2]],
                                            &transformed2[tess_faces[i][3]]);
            }
        }

        // 绘制两个超立方体的边框
        for (uint i = 0; i < count_of(tess_edges); ++i) {
            pixel_t col = colours[i % count_of(colours)];
            graphics_draw_line(volume, &transformed1[tess_edges[i][0]],
                                       &transformed1[tess_edges[i][1]], col);
            graphics_draw_line(volume, &transformed2[tess_edges[i][0]],
                                       &transformed2[tess_edges[i][1]], col);
        }

        voxel_buffer_swap();
        usleep(50000);
        frame++;
    }

    voxel_buffer_unmap();
    return 0;
}