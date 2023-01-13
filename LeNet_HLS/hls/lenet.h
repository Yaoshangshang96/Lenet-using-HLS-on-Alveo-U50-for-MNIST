#ifndef CONV_NET_H
#define CONV_NET_H

#include <stdint.h>

typedef int32_t DTYPE;
typedef int32_t PARA_DTYPE;

const uint8_t IMG_SIZE = 28;

const uint8_t C1_OUT_SIZE = 12;
const uint8_t C1_OUT_CHANNELS = 16;
const uint8_t C1_FLITER_SIZE = 5;
const uint8_t C1_STRIDE = 2;

const uint8_t C2_OUT_SIZE = 4;
const uint8_t C2_OUT_CHANNELS = 16;
const uint8_t C2_FLITER_SIZE = 3;
const uint8_t C2_STRIDE = 1;

void lenet_top(uint8_t *in, uint8_t *out);
void lenet(DTYPE img[IMG_SIZE][IMG_SIZE], uint8_t p[1]);
void conv_layer_1(
		DTYPE in[IMG_SIZE][IMG_SIZE],
		const PARA_DTYPE weight[C1_OUT_CHANNELS][C1_FLITER_SIZE][C1_FLITER_SIZE],
		const PARA_DTYPE bias[C1_OUT_CHANNELS],
		DTYPE out[C1_OUT_CHANNELS][C1_OUT_SIZE][C1_OUT_SIZE]
);

void relu_layer_1(
		DTYPE in[C1_OUT_CHANNELS][C1_OUT_SIZE][C1_OUT_SIZE],
		DTYPE out[C1_OUT_CHANNELS][C1_OUT_SIZE][C1_OUT_SIZE]
);

void pooling_layer_1(
		DTYPE in[C1_OUT_CHANNELS][C1_OUT_SIZE][C1_OUT_SIZE],
		DTYPE out[C1_OUT_CHANNELS][C1_OUT_SIZE/2][C1_OUT_SIZE/2]
);

void conv_layer_2(
		DTYPE in[C1_OUT_CHANNELS][C1_OUT_SIZE/2][C1_OUT_SIZE/2],
		const PARA_DTYPE weight[C2_OUT_CHANNELS][C1_OUT_CHANNELS][C2_FLITER_SIZE][C2_FLITER_SIZE],
		const PARA_DTYPE bias[C2_OUT_CHANNELS],
		DTYPE out[C2_OUT_CHANNELS][C2_OUT_SIZE][C2_OUT_SIZE]
);

void relu_layer_2(
		DTYPE in[C2_OUT_CHANNELS][C2_OUT_SIZE][C2_OUT_SIZE],
		DTYPE out[C2_OUT_CHANNELS][C2_OUT_SIZE][C2_OUT_SIZE]
);

void pooling_layer_2(
		DTYPE in[C2_OUT_CHANNELS][C2_OUT_SIZE][C2_OUT_SIZE],
		DTYPE out[C2_OUT_CHANNELS][C2_OUT_SIZE/2][C2_OUT_SIZE/2]
);

void flatten_layer(
		DTYPE in[C2_OUT_CHANNELS][C2_OUT_SIZE/2][C2_OUT_SIZE/2],
		DTYPE out[(C2_OUT_SIZE/2) * (C2_OUT_SIZE/2) * C2_OUT_CHANNELS]
);

void full_connection_layer(
		DTYPE in[(C2_OUT_SIZE/2) * (C2_OUT_SIZE/2) * C2_OUT_CHANNELS],
		const PARA_DTYPE weight[10][(C2_OUT_SIZE/2) * (C2_OUT_SIZE/2) * C2_OUT_CHANNELS],
		const PARA_DTYPE bias[10],
		DTYPE out[10]
);

void softmax(
		DTYPE in[10],
		uint8_t out[1]
);

#endif
