<script setup lang="ts">
import { ref, watchEffect } from 'vue'
import type { Ref } from 'vue'
import { Experiment } from '@/experiment'

const API_URL = "http://linuxtv/nn-api"

const experiments: Ref<Array<Experiment>> = ref(new Array())

watchEffect(async () => {
  const url = `${API_URL}/experiments`

  const expsIn = await (await fetch(url)).json()
  for (const expIn of expsIn) {
    const exp = Experiment.from_json(expIn)
    experiments.value.push(exp)
  }
})
</script>


<template>
  <div class="experiments">
    <span class="header shortcode">shortcode</span>
    <span class="header nepochs">nepochs</span>
    <span class="header net_class">net_class</span>

    <template v-for="exp in experiments">
      <span class="experiment">
        <span class="shortcode">{{ exp.shortcode }}</span>
        <span class="nepochs">{{ exp.nepochs }}</span>
        <span class="net_class">{{ exp.net_class }}</span>
      </span>
    </template>
  </div>
</template>

<style>
.experiments {
    display: grid;
    width: max-content;
}

.experiments .header {
    margin-right: 10px;
}
.experiments .experiment {
    display: contents;
}

.experiments .shortcode {
    grid-column: 1;
}

.experiments .nepochs {
    grid-column: 2;
    width: auto;
}

.experiments .net_class {
    grid-column: 3;
    width: auto;
}
</style>