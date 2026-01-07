// nasm -f elf64 -O3 l2_kernel.asm -o l2_kernel.o
// clang -std=c17 -O3 -march=native -fno-math-errno -fno-trapping-math -flto
// -mtune=native -mavx2 -mfma main.c l2_kernel.o -o spann_demo -lm

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Assembly Prototype for high-performance L2 distance */
extern float l2_sq_dist_avx2(uint32_t dim, const float *a, const float *b);

typedef struct {
  float *coordinates;
  uint32_t dimension;
} vector_t;

typedef struct {
  vector_t centroid;
  uint64_t disk_offset;
  uint32_t list_size;
} posting_meta_t;

typedef struct {
  uint32_t dimension;
  uint32_t num_centroids;
  posting_meta_t *centroids;
  FILE *posting_file;
} spann_index_t;

typedef struct {
  uint32_t id;
  float distance;
} neighbor_t;

/**
 * Selects the most promising posting lists based on the query vector.
 * Limits disk access to the 'top' candidates to maintain low latency.
 */
static size_t select_posting_lists(const spann_index_t *index,
                                   const vector_t *query, const float epsilon_2,
                                   uint32_t *active_indices,
                                   float *centroid_dists,
                                   const size_t max_candidates) {
  if (index->num_centroids == 0) {
    return 0;
  }

  float min_dist = INFINITY;
  for (uint32_t i = 0; i < index->num_centroids; ++i) {
    float d = l2_sq_dist_avx2(index->dimension, query->coordinates,
                              index->centroids[i].centroid.coordinates);
    centroid_dists[i] = d;
    if (d < min_dist) {
      min_dist = d;
    }
  }

  float threshold = (1.0f + epsilon_2) * (1.0f + epsilon_2) * min_dist;
  size_t active_count = 0;

  for (uint32_t i = 0;
       i < index->num_centroids && active_count < max_candidates; ++i) {
    if (centroid_dists[i] <= threshold) {
      active_indices[active_count++] = i;
    }
  }
  return active_count;
}

/**
 * SPANN Search Core: Scans disk-resident vectors within selected clusters.
 */
neighbor_t search_spann(const spann_index_t *index, const vector_t *query,
                        const float epsilon_2) {
  neighbor_t best_neighbor = {0, INFINITY};

  /* Allocate temporary storage for centroid distances to avoid VLA usage */
  float *centroid_dists = malloc((size_t)index->num_centroids * sizeof(float));
  if (!centroid_dists) {
    perror("Failed to allocate centroid distance buffer");
    return best_neighbor;
  }

  uint32_t active_lists[32];
  /* Limit to top 32 lists for search efficiency */
  size_t num_lists = select_posting_lists(index, query, epsilon_2, active_lists,
                                          centroid_dists, 32);
  free(centroid_dists);

  const uint32_t block_vectors = 256;
  const size_t block_floats = (size_t)block_vectors * index->dimension;
  const size_t buffer_size = block_floats * sizeof(float);
  const size_t alignment = 32;

  /* * C17 aligned_alloc: size must be a multiple of alignment.
   * 32-byte alignment is required for AVX2.
   * buffer_size is (256 * dim * 4). If dim is even, buffer_size % 32 == 0.
   * We assume dim=128 from main, so 128*256*4 = 131072, which is valid.
   */
  float *buffer = aligned_alloc(alignment, buffer_size);
  if (!buffer) {
    /* Fallback to standard malloc if aligned_alloc fails (rare) */
    buffer = malloc(buffer_size);
  }
  if (!buffer) {
    perror("Failed to allocate vector buffer");
    return best_neighbor;
  }

  for (size_t i = 0; i < num_lists; ++i) {
    const posting_meta_t *meta = &index->centroids[active_lists[i]];
    const uint64_t list_base_id =
        meta->disk_offset / (index->dimension * sizeof(float));

    /* * Note: fseek takes a 'long'. On LLP64 (Windows), this is 32-bit (2GB
     * max). On LP64 (Linux), it is 64-bit. Since input uses ELF64/AVX, we
     * assume LP64.
     */
    if (fseek(index->posting_file, (long)meta->disk_offset, SEEK_SET) != 0) {
      perror("Seek failed");
      continue;
    }

    uint32_t remaining = meta->list_size;
    uint32_t list_offset = 0;
    while (remaining > 0) {
      uint32_t read_vectors =
          remaining > block_vectors ? block_vectors : remaining;

      size_t read_count =
          fread(buffer, sizeof(float), (size_t)read_vectors * index->dimension,
                index->posting_file);

      if (read_count != (size_t)read_vectors * index->dimension) {
        break;
      }

      for (uint32_t j = 0; j < read_vectors; ++j) {
        float *vec = buffer + (size_t)j * index->dimension;
        float d = l2_sq_dist_avx2(index->dimension, query->coordinates, vec);
        if (d < best_neighbor.distance) {
          best_neighbor.distance = d;
          best_neighbor.id = (uint32_t)(list_base_id + list_offset + j);
        }
      }

      remaining -= read_vectors;
      list_offset += read_vectors;
    }
  }

  free(buffer);
  return best_neighbor;
}

void destroy_index(spann_index_t *index) {
  if (!index) {
    return;
  }
  if (index->centroids) {
    for (uint32_t i = 0; i < index->num_centroids; ++i) {
      free(index->centroids[i].centroid.coordinates);
    }
    free(index->centroids);
  }
  if (index->posting_file) {
    fclose(index->posting_file);
  }
  free(index);
}

int main(void) {
  /* SCALE: 5 Million Vectors */
  /* File size will be approx 2.56 GB (128 dims * 4 bytes * 5M) */
  const uint32_t dim = 128;
  const uint32_t num_centroids = 100;
  const uint32_t vectors_per_list = 50000;
  const char *posting_filename = "spann_5m_data.bin";

  printf("--- SPANN Optimized (AVX2) 5M Vector Demo ---\n");
  printf("Target: 5,000,000 Vectors\n");
  printf("Estimated Disk Usage: ~2.56 GB\n\n");

  spann_index_t *index = calloc(1, sizeof(spann_index_t));
  if (!index) {
    perror("Failed to allocate index");
    return 1;
  }

  index->dimension = dim;
  index->num_centroids = num_centroids;
  index->centroids = calloc(num_centroids, sizeof(posting_meta_t));
  if (!index->centroids) {
    perror("Failed to allocate centroids");
    free(index);
    return 1;
  }

  /* Check if the file already exists */
  FILE *exists_check = fopen(posting_filename, "rb");
  bool file_exists = (exists_check != NULL);
  if (file_exists) {
    fclose(exists_check);
    printf("Detected existing data file: %s. Loading metadata...\n",
           posting_filename);
    index->posting_file = fopen(posting_filename, "rb+");
  } else {
    printf("Data file not found. Generating synthetic data and streaming to "
           "disk...\n");
    index->posting_file = fopen(posting_filename, "wb+");
  }

  if (!index->posting_file) {
    perror("Failed to open file. Check disk space/permissions.");
    destroy_index(index);
    return 1;
  }

  /* Optimize I/O buffer */
  if (setvbuf(index->posting_file, NULL, _IOFBF, 4 * 1024 * 1024) != 0) {
    perror("Warning: Failed to set file buffer");
  }

  srand(42);
  const float inv_rand_max = 1.0f / (float)RAND_MAX;
  float *temp_vec = malloc(dim * sizeof(float));
  if (!temp_vec) {
    perror("Failed to allocate temp_vec");
    destroy_index(index);
    return 1;
  }

  /* Initialization logic */
  for (uint32_t i = 0; i < num_centroids; ++i) {
    index->centroids[i].centroid.coordinates = malloc(dim * sizeof(float));
    if (!index->centroids[i].centroid.coordinates) {
      perror("Failed to allocate centroid coordinates");
      /* Simple cleanup attempt, actual recovery is complex here */
      free(temp_vec);
      destroy_index(index);
      return 1;
    }

    index->centroids[i].list_size = vectors_per_list;
    index->centroids[i].disk_offset =
        (uint64_t)i * vectors_per_list * dim * sizeof(float);

    /* Deterministic random seed for centroids so they match across runs */
    srand(i);
    for (uint32_t d = 0; d < dim; d++) {
      index->centroids[i].centroid.coordinates[d] =
          (float)rand() * inv_rand_max;
    }

    /* Only write vectors if the file didn't exist */
    if (!file_exists) {
      if (fseek(index->posting_file, (long)index->centroids[i].disk_offset,
                SEEK_SET) != 0) {
        perror("File seek error during generation");
        break;
      }

      for (uint32_t v = 0; v < vectors_per_list; v++) {
        for (uint32_t d = 0; d < dim; d++) {
          temp_vec[d] = index->centroids[i].centroid.coordinates[d] +
                        ((float)rand() * inv_rand_max * 0.02f);
        }

        if (fwrite(temp_vec, sizeof(float), dim, index->posting_file) != dim) {
          perror("File write error");
          break;
        }
      }

      if (i % 10 == 0) {
        printf("\rProgress: %.1f%% (%u/%u centroids written)",
               ((float)i / num_centroids) * 100.0f, i, num_centroids);
        fflush(stdout);
      }
    }
  }

  if (!file_exists) {
    printf("\nData Generation Complete.\n");
  } else {
    printf("Metadata loaded successfully. Skipping data generation.\n");
  }

  fflush(index->posting_file);
  free(temp_vec);

  /* Create Query */
  srand((unsigned int)time(NULL));
  vector_t query;
  query.dimension = dim;
  query.coordinates = malloc(dim * sizeof(float));

  if (query.coordinates) {
    for (uint32_t d = 0; d < dim; d++) {
      query.coordinates[d] = (float)rand() * inv_rand_max;
    }

    printf("\nExecuting search against 5M vectors (Disk-Resident)...\n");

    clock_t start = clock();
    neighbor_t result = search_spann(index, &query, 0.3f);
    clock_t end = clock();

    double latency_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    printf("\nSearch Results:\n");
    printf("Nearest Neighbor ID: %u\n", result.id);
    printf("L2 Distance:         %f\n", result.distance);
    printf("Total Latency:       %.3f ms\n", latency_ms);

    free(query.coordinates);
  } else {
    perror("Failed to allocate query vector");
  }

  destroy_index(index);

  return 0;
}
