#include "lenet.h"
#include "parameters.h"
#include <stdio.h>
#include <string.h>


void lenet_top(uint8_t *in, uint8_t *out)  {

	#pragma HLS INTERFACE s_axilite port=in
	#pragma HLS INTERFACE m_axi port=in offset=slave
	#pragma HLS INTERFACE s_axilite port=out
	#pragma HLS INTERFACE m_axi port=out offset=slave
	#pragma HLS INTERFACE s_axilite port=return

	DTYPE img[IMG_SIZE][IMG_SIZE];
	uint8_t p[1];

	uint8_t in_buffer[IMG_SIZE * IMG_SIZE] ;

	memcpy(in_buffer,in,IMG_SIZE * IMG_SIZE * sizeof(uint8_t));
	uint16_t x = 0;
	for(uint16_t i=0; i < IMG_SIZE;i++){
		for(uint16_t j=0; j < IMG_SIZE;j++){

		img [i][j] = in_buffer[x];

		x++;
		}
	}

    lenet(img,p);
	out[0] = p[0];
}

void lenet(DTYPE img[IMG_SIZE][IMG_SIZE], uint8_t p[1]){

	DTYPE conv_layer1_out[C1_OUT_CHANNELS][C1_OUT_SIZE][C1_OUT_SIZE];
	DTYPE relu_layer1_out[C1_OUT_CHANNELS][C1_OUT_SIZE][C1_OUT_SIZE];
	DTYPE pooling_layer1_out[C1_OUT_CHANNELS][C1_OUT_SIZE/2][C1_OUT_SIZE/2];

	DTYPE conv_layer2_out[C2_OUT_CHANNELS][C2_OUT_SIZE][C2_OUT_SIZE];
	DTYPE relu_layer2_out[C2_OUT_CHANNELS][C2_OUT_SIZE][C2_OUT_SIZE];
	DTYPE pooling_layer2_out[C2_OUT_CHANNELS][C2_OUT_SIZE/2][C2_OUT_SIZE/2];

	DTYPE flatten_layer_out[(C2_OUT_SIZE/2)*(C2_OUT_SIZE/2)*C2_OUT_CHANNELS];
	DTYPE full_connection_layer_out[10];

	conv_layer_1(img,weight_l1,bias_l1,conv_layer1_out);
	relu_layer_1(conv_layer1_out,relu_layer1_out);
	pooling_layer_1(relu_layer1_out,pooling_layer1_out);
	conv_layer_2(pooling_layer1_out,weight_l2,bias_l2,conv_layer2_out);
	relu_layer_2(conv_layer2_out,relu_layer2_out);
	pooling_layer_2(relu_layer2_out,pooling_layer2_out);
	flatten_layer(pooling_layer2_out,flatten_layer_out);
	full_connection_layer(flatten_layer_out,weight_l3,bias_l3,full_connection_layer_out);
	softmax(full_connection_layer_out,p);
}


void conv_layer_1(
		DTYPE in[IMG_SIZE][IMG_SIZE],
		const PARA_DTYPE weight[C1_OUT_CHANNELS][C1_FLITER_SIZE][C1_FLITER_SIZE],
		const PARA_DTYPE bias[C1_OUT_CHANNELS],
		DTYPE out[C1_OUT_CHANNELS][C1_OUT_SIZE][C1_OUT_SIZE]
){
	for(uint8_t i = 0;i < C1_OUT_CHANNELS;i++){
		for(uint8_t j = 0;j < C1_OUT_SIZE;j++){
			for(uint8_t k = 0;k < C1_OUT_SIZE;k++){
				#pragma hls unroll
				out[i][j][k] = bias[i];
				}
			}
	}
		for(uint8_t i = 0;i < C1_OUT_CHANNELS;i++){
			for(uint8_t j = 0;j < IMG_SIZE - C1_FLITER_SIZE + 1;j+=C1_STRIDE){
				for(uint8_t k = 0;k < IMG_SIZE - C1_FLITER_SIZE + 1;k+=C1_STRIDE){
					for(uint8_t r = 0;r < C1_FLITER_SIZE;r++){
						for(uint8_t c = 0;c < C1_FLITER_SIZE;c++){
								#pragma hls unroll
								out[i][j/C1_STRIDE][k/C1_STRIDE] += weight[i][r][c] * in[j+r][k+c];
								}
						}
					}
				}
		}
}

void relu_layer_1(
		DTYPE in[C1_OUT_CHANNELS][C1_OUT_SIZE][C1_OUT_SIZE],
		DTYPE out[C1_OUT_CHANNELS][C1_OUT_SIZE][C1_OUT_SIZE]
){

	for(uint8_t i = 0;i < C1_OUT_CHANNELS;i++){
		for(uint8_t j = 0;j < C1_OUT_SIZE;j++){
			for(uint8_t k = 0;k < C1_OUT_SIZE;k++){
				#pragma hls unroll
				out[i][j][k] = (in[i][j][k] < 0)?0 : (in[i][j][k]>>6);//
				}
			}
	}
}

void pooling_layer_1(
		DTYPE in[C1_OUT_CHANNELS][C1_OUT_SIZE][C1_OUT_SIZE],
		DTYPE out[C1_OUT_CHANNELS][C1_OUT_SIZE/2][C1_OUT_SIZE/2]
){
	DTYPE temp0,temp1;
	for(uint8_t i = 0;i < C1_OUT_CHANNELS;i++){
			for(uint8_t j = 0;j < C1_OUT_SIZE/2;j++){
				for(uint8_t k = 0;k < C1_OUT_SIZE/2;k++){
					#pragma hls unroll
					temp0 = (in[i][j*2][k*2]     > in[i][j*2 + 1][k*2]    )?in[i][j*2][k*2]    :in[i][j*2 + 1][k*2]    ;
					temp1 = (in[i][j*2][k*2 + 1] > in[i][j*2 + 1][k*2 + 1])?in[i][j*2][k*2 + 1]:in[i][j*2 + 1][k*2 + 1];
					out[i][j][k] = (temp0 > temp1)?temp0:temp1;
					}
				}
		}
}

void conv_layer_2(
		DTYPE in[C1_OUT_CHANNELS][C1_OUT_SIZE/2][C1_OUT_SIZE/2],
		const PARA_DTYPE weight[C2_OUT_CHANNELS][C1_OUT_CHANNELS][C2_FLITER_SIZE][C2_FLITER_SIZE],
		const PARA_DTYPE bias[C2_OUT_CHANNELS],
		DTYPE out[C2_OUT_CHANNELS][C2_OUT_SIZE][C2_OUT_SIZE]
){
	for(uint8_t i = 0;i < C2_OUT_CHANNELS;i++){
		for(uint8_t j = 0;j < C2_OUT_SIZE;j++){
			for(uint8_t k = 0;k < C2_OUT_SIZE;k++){
				#pragma hls unroll
				out[i][j][k] = bias[i];
				}
			}
	}
	for (uint8_t l = 0;l < C1_OUT_CHANNELS;l++){
		for(uint8_t i = 0;i < C2_OUT_CHANNELS;i++){
			for(uint8_t j = 0;j < C1_OUT_SIZE/2 - C2_FLITER_SIZE + 1;j+=C2_STRIDE){
				for(uint8_t k = 0;k < C1_OUT_SIZE/2 - C2_FLITER_SIZE + 1;k+=C2_STRIDE){
					for(uint8_t r = 0;r < C2_FLITER_SIZE;r++){
						for(uint8_t c = 0;c < C2_FLITER_SIZE;c++){
							#pragma hls unroll
							out[i][j][k] += weight[i][l][r][c] * in[l][j+r][k+c];
						}
					}
				}
			}
		}
	}
}

void relu_layer_2(
		DTYPE in[C2_OUT_CHANNELS][C2_OUT_SIZE][C2_OUT_SIZE],
		DTYPE out[C2_OUT_CHANNELS][C2_OUT_SIZE][C2_OUT_SIZE]
){
	for(uint8_t i = 0;i < C2_OUT_CHANNELS;i++){
		for(uint8_t j = 0;j < C2_OUT_SIZE;j++){
			for(uint8_t k = 0;k < C2_OUT_SIZE;k++){
#pragma hls unroll
				out[i][j][k] = (in[i][j][k] < 0)?0 : (in[i][j][k]>>6);//
				}
			}
	}
}

void pooling_layer_2(
		DTYPE in[C2_OUT_CHANNELS][C2_OUT_SIZE][C2_OUT_SIZE],
		DTYPE out[C2_OUT_CHANNELS][C2_OUT_SIZE/2][C2_OUT_SIZE/2]
){
	DTYPE temp2,temp3;
	for(uint8_t i = 0;i < C2_OUT_CHANNELS;i++){
			for(uint8_t j = 0;j < C2_OUT_SIZE/2;j++){
				for(uint8_t k = 0;k < C2_OUT_SIZE/2;k++){
#pragma hls unroll
					temp2 = (in[i][j*2][k*2]     > in[i][j*2 + 1][k*2]    )?in[i][j*2][k*2]    :in[i][j*2 + 1][k*2]    ;
					temp3 = (in[i][j*2][k*2 + 1] > in[i][j*2 + 1][k*2 + 1])?in[i][j*2][k*2 + 1]:in[i][j*2 + 1][k*2 + 1];
					out[i][j][k] = (temp2 > temp3)?temp2:temp3;
					}
				}
		}
}

void flatten_layer(
		DTYPE in[C2_OUT_CHANNELS][C2_OUT_SIZE/2][C2_OUT_SIZE/2],
		DTYPE out[(C2_OUT_SIZE/2) * (C2_OUT_SIZE/2) * C2_OUT_CHANNELS]
){
		uint16_t t = 0;
	for(uint8_t i = 0;i < C2_OUT_CHANNELS;i++){
		for(uint8_t j = 0;j < C2_OUT_SIZE/2;j++){
			for(uint8_t k = 0;k < C2_OUT_SIZE/2;k++){

#pragma hls unroll
					out[t++] = in[i][j][k];

			}
		}
	}
}

void full_connection_layer(
		DTYPE in[(C2_OUT_SIZE/2) * (C2_OUT_SIZE/2) * C2_OUT_CHANNELS],
		const PARA_DTYPE weight[10][(C2_OUT_SIZE/2) * (C2_OUT_SIZE/2) * C2_OUT_CHANNELS],
		const PARA_DTYPE bias[10],
		DTYPE out[10]
){
	for(uint8_t i = 0;i < 10;i++){
#pragma hls unroll
		out[i] = bias[i];
	}
	for(uint8_t k = 0;k < 10;k++){
	for(uint32_t j = 0;j < (C2_OUT_SIZE/2) * (C2_OUT_SIZE/2) * C2_OUT_CHANNELS;j++){

#pragma hls unroll
					out[k] += in[j] * weight[k][j];
				}
	}
}

void softmax(
		DTYPE in[10],
		uint8_t out[1]
){
	DTYPE max_value = 0;
	uint8_t max_i = 0;

	for (uint8_t i = 0; i<10 ;i++){
#pragma hls unroll
		if(in[i] > max_value){
		max_value = in[i];
		max_i = i;
		}
	}
	out[0] = max_i;
}
