#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

// Initialize the LED matrix display
void led_matrix_init(void);

// You may need to add more functions here
void led_matrix_display_char(char c);

void led_matrix_display_string(const char *str, bool marquee, int num_repeats);

void update_led_states(bool new_states[5][5]);