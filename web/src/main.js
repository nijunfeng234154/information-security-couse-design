import './assets/main.css'

import { createApp } from 'vue'
import { createPinia } from 'pinia'

import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'

import App from './App.vue'
import router from './router'
import TopBar from './views/TopBar.vue'

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(ElementPlus)

app.component('TopBar', TopBar)
app.mount('#app')
