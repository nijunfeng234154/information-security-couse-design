<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import axios from 'axios'

const imageUrl = ref('')
const NoiseImageUrl = ref('')
const carrierImageUrl = ref('')
const uploadRef = ref()
const uploadFile = ref()
const psnr = ref(0)
const resultImageUrl = ref('')
const resultMsg = ref('')
const showSidebar = ref(false)
const input1 = ref('')
const input2 = ref('')
const decrypt = ref(false)
const taskLogs = ref([])
const choice = ref('')

const clearOutputImages = () => {
  NoiseImageUrl.value = null
  carrierImageUrl.value = null
  resultImageUrl.value = null
}

//前端日志记录
const addLog = (log) => {
  taskLogs.value.push(log)
  if (taskLogs.value.length > 10) {
    taskLogs.value.shift()
  }
}

const handleAvatarSuccess = (
  response,
  uploadFile
) => {
  imageUrl.value = URL.createObjectURL(uploadFile.raw)
}

const handleImgChange = (file) => {
  imageUrl.value = URL.createObjectURL(file.raw)
  uploadFile.value = file
}

const beforeAvatarUpload = (rawFile) => {
  clearOutputImages()
  if (rawFile.type !== 'image/jpeg/png/jpg') {
    ElMessage.error('Avatar picture must be image format!')
    return false
  }
  return true
}

//处理生成载体图像请求,以及揭示请求
const handleSubmit = async (decrypt) => {
  clearOutputImages()
  if (!uploadFile.value || !uploadFile.value.raw) {
    ElMessage.error('请先选择一个文件')
    return
  }

  const file = uploadFile.value.raw
  const reader = new FileReader()

  reader.onload = async (e) => {
    let imgbase64 = e.target.result
    // 如果包含图片类型和Base64的前缀，去掉它
    if (imgbase64.startsWith('data:image')) {
      imgbase64 = imgbase64.split(';base64,').pop() // 只保留base64编码部分
    }

    console.log("发送图片:: ", input1.value, input2.value, decrypt)
    addLog(`发送图片`)
    addLog(`${input1.value} ${input2.value}`) 
    addLog('是否为揭示模式：'+`${decrypt}`)
    addLog('开始解析请求')
    axios.post('http://localhost:5000/encrypt', {
      image: imgbase64,
      key: input1.value,
      pub: input2.value,
      decrypt: decrypt
    }, {
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    }).then(async res => {
      console.log("请求完成")
      addLog('请求完成')
      if (res.data && res.data.data) {
        addLog("JSON参数解析完毕，开始转换图片")
        let data = res.data.data
        let status = res.data.status
        let psnr = res.data.psnr
        if (status === 'error') {
          addLog('Base64解码失败:', res.data.code)
          return
        } else {
          if (decrypt) {
            addLog('揭示成功')
            addLog('恢复出图像和原始图像的峰值信噪比:'+psnr)
          } else {
            addLog('隐藏成功')
          }
        }
        let base64Image = data.split(';base64,').pop()
        let blob = await fetch(`data:image/jpeg;base64,${base64Image}`).then(response => response.blob())
        resultImageUrl.value = URL.createObjectURL(blob)
        if (decrypt === false) {
          carrierImageUrl.value = resultImageUrl.value
        } else {
          // NoiseImageUrl.value = imageUrl.value
        }
        console.log(resultImageUrl.value)
      } else {
        ElMessage.error('生成载体图像失败-Invalid response data:', res.data)
        addLog('生成载体图像失败-Invalid response data:', res.data)
      }
    })
  }

  reader.readAsDataURL(file)
}

const toggleSidebar = () => {
  showSidebar.value = !showSidebar.value
}
//处理图像噪声模拟请求
const handleNoise = async (choice) => {
  if (!uploadFile.value || !uploadFile.value.raw) {
    ElMessage.error('请先选择一个文件')
    return
  }

  const file = uploadFile.value.raw
  const reader = new FileReader()

  reader.onload = async (e) => {
    let imgbase64 = e.target.result
    // 如果包含图片类型和Base64的前缀，去掉它
    if (imgbase64.startsWith('data:image')) {
      imgbase64 = imgbase64.split(';base64,').pop() // 只保留base64编码部分
    }

    addLog(`图片增加噪声:`)
    addLog('开始解析请求')
    if(choice === '1'){
      addLog('收到加噪请求：压缩图像')
    }
    if(choice === '2'){
      addLog('收到加噪请求：裁剪图像')
    }
    if(choice === '3'){
      addLog('收到加噪请求：加噪声')
      addLog('随机高斯噪声、椒盐噪声')
    }
    if(choice === '4'){
      addLog('收到加噪请求：光线变化')
    }
    addLog('正在处理...')
    axios.post('http://localhost:5000/noise', {
      image: imgbase64,
      choice: choice
    }, {
      headers: {
        'Content-Type': 'application/json',
      }
    }).then(async res => {
      if (res.data && res.data.data) {
        addLog("请求完成")
        let data = res.data.data
        let status = res.data.status
        if (status === 'error') {
          addLog('Base64解码失败:', res.data.code)
          return
        } else {
          addLog('加噪声成功')
        }
        let base64Image = data.split(';base64,').pop()
        let blob = await fetch(`data:image/jpeg;base64,${base64Image}`).then(response => response.blob())
        NoiseImageUrl.value = URL.createObjectURL(blob)
        console.log(NoiseImageUrl.value)
      } else {
        ElMessage.error('生成加噪图像失败-Invalid response data:', res.data)
        addLog('生成加噪图像失败-Invalid response data:', res.data)
      }
    })
}
  reader.readAsDataURL(file)
}

const clearAll = () => {
  clearOutputImages()
  imageUrl.value = null
}

</script>

<template>
  <main>
    <!--左边竖放进度条 -->
    <!-- <el-progress v-if="currentTaskId" :percentage="progressPercentage" stroke-width="18" :status="progressStatus"> -->
    <!--</el-progress> -->
    <div style="display: flex;justify-content:flex-start;flex-flow:column;">
      <!--      上半部分三个框，第一个框是上传框-->
      <div class="title">
        <el-span>原始图像上传</el-span>
        <el-span>载体图像生成</el-span>
        <el-span>加噪图像生成</el-span>
      </div>
      <div class="top">
        <el-row>
          <el-upload class="avatar-uploader" ref="uploadRef"
            action="https://run.mocky.io/v3/9d059bf9-4660-45f2-925d-ce80ad6c4d15" :show-file-list="false"
            :on-success="handleAvatarSuccess" :before-upload="beforeAvatarUpload" :auto-upload="false"
            :on-change="handleImgChange">
            <img v-if="imageUrl" :src="imageUrl" class="avatar" alt="" style="" />
            <el-icon v-else class="avatar-uploader-icon">
              <Plus />
            </el-icon>
          </el-upload>
        </el-row>
        <el-row>
          <div class="receiveImg">
            <img v-if="carrierImageUrl" class="receiveImg" :src="carrierImageUrl" alt="" />
          </div>
        </el-row>
        <el-row>
          <div class="receiveImg">
            <img v-if="NoiseImageUrl" class="receiveImg" :src="NoiseImageUrl" alt="" />
          </div>
        </el-row>
      </div>
      <div class="medium" style="display: flex;justify-content: flex-start;margin: 20px 0;">
        <el-row>
          <el-input v-model="input1" style="width:178px; margin-right:16px" placeholder="输入私钥" />
        </el-row>
        <el-row>
          <el-input v-model="input2" style="width:178px; margin-right:16px" placeholder="输入公钥" />
        </el-row>
        <el-row>
          <el-button style="margin-top:3px" @click="toggleSidebar">噪声模拟</el-button>
        </el-row>
        <el-col>
          <div v-if="showSidebar" class="sidebar">
            <el-button @click="handleNoise('1')">压缩</el-button>
            <el-button @click="handleNoise('2')">裁剪</el-button>
            <el-button @click="handleNoise('3')">加噪声</el-button>
            <el-button @click="handleNoise('4')">光线变化</el-button>
          </div>
        </el-col>
      </div>
      <div class="bottom" style="display:flex;justify-content:flex-start;margin-right:200px;">
        <div class="bottomleft">
          <!-- 日志框 -->
          <div class="log-box">
            <h3>logs</h3>
            <div class="log-content">
              <p v-for="(log, index) in taskLogs" :key="index">{{ log }}</p>
            </div>
          </div>
        </div>
        <div class="bottomright"
          style="display: flex; justify-content: space-evenly; margin-left:10px; flex-flow:column; width:186px;">
          <div class="receiveImg">
            <img v-if="resultImageUrl" class="receiveImg" :src="resultImageUrl" alt="" />
          </div>
          <div class="bottombuttons">
            <el-button style="width:100px" @click="handleSubmit(false)">获取载体图像</el-button>
            <el-button style="width:100px" @click="handleSubmit(true)">获取原始图像</el-button>
            <el-button style="width:100px" @click="clearAll">清空所有图片</el-button>
          </div>
          <!--        <div style="margin-top: 12px">-->
          <!--          <el-text v-text="resultMsg" style="justify-content: center;font-size: 18px" class="mx-1" type="success" szie="large" :src="resultMsg"></el-text>-->
          <!--        </div>-->
        </div>
      </div>
    </div>
  </main>
</template>

<style scoped>
* {
  color: aquamarine;
}

.title {
  display: flex;
  justify-content: flex-start;
  margin-bottom: 5px;
  margin-left: 44px;
}

.title>* {
  margin-right: 104px;
}

main {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  overflow: hidden;
}

.avatar-uploader .avatar {
  width: 178px;
  height: 178px;
  display: block;
}

.top {
  display: flex;
  justify-content: flex-start;
  align-items: center;
}

.top> :nth-child(2) {
  margin-right: 16px;
}

.top> :nth-child(1) {
  margin-right: 16px;
}

.medium> :nth-child(1),
.medium> :nth-child(2) {
  margin-right: 16px;
  width: 178px;
  padding: 0;
}

.medium .el-button {
  height: 38px;
}

.log-box {
  display: flex;
  flex-flow: column;
  justify-content: flex-start;
  align-items: center;
  width: 186px;
  height: 178px;
  border-radius: 5px;
}

.bottombuttons {
  display: flex;
  justify-content: space-evenly;
  margin-top: 6px;
  margin-left: 39px;
  flex-flow: column;
}

.bottombuttons>* {
  margin: 3px 0 0 0 !important;
}
</style>

<style>
.medium {
  height: 40px;
}

.sidebar {
  display: flex;
  flex-flow: column;
  background-color: #333;
  width: 80px;
  color: white;
  margin-left: 10px;
  border-radius: 5px;
}

.sidebar>* {
  margin: 3px 0 0 0 !important;
}

.avatar-uploader .el-upload {
  border: 1px dashed var(--el-border-color);
  border-radius: 6px;
  cursor: pointer;
}

.avatar-uploader .el-upload:hover {
  border-color: var(--el-color-primary);
}

.el-icon.avatar-uploader-icon {
  font-size: 28px;
  color: #8c939d;
  width: 178px;
  height: 178px;
  text-align: center;
}

.receiveImg {
  width: 178px;
  height: 178px;
  display: block;
  border: 1px dashed var(--el-border-color);
  border-radius: 6px;
}

.log-content {
  display: flex;
  width: 178px;
  height: 178px;
  flex-flow: column;
  overflow-y: auto;
  font-size: 12px;
  border: 1px dashed var(--el-border-color);
  border-radius: 6px;
}
</style>
