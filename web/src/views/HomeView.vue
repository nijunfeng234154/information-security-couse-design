<script setup>

import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import axios from 'axios'

const imageUrl = ref('')
const uploadRef = ref()
const uploadFile = ref()
const input = ref('')
const resultImageUrl = ref('')
const resultMsg = ref('')
const showSidebar = ref(false)
const input1 = ref('')
const input2 = ref('')
const decrypt = ref(false)


const clearAllImages = () => {
    NoiseImageUrl.value = null
    carrierImageUrl.value = null
    resultImageUrl.value = null
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
  clearAllImages()
  if (rawFile.type !== 'image/jpeg/png/jpg') {
    ElMessage.error('Avatar picture must be image format!')
    return false
  }
  return true
}

//处理生成载体图像请求,以及解密请求
const handleSubmit = async (decrypt) => {
  clearAllImages()
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
    axios.post('http://127.0.0.1:5000/encrypt', {
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
      if (res.data && res.data.data) {
        let data = res.data.data
        let base64Image = data.split(';base64,').pop()
        let blob = await fetch(`data:image/jpeg;base64,${base64Image}`).then(response => response.blob())
        resultImageUrl.value = URL.createObjectURL(blob)
        if (decrypt===false){
          carrierImageUrl.value = resultImageUrl.value
        } else {
          NoiseImageUrl.value = imageUrl.value
        }
        console.log(resultImageUrl.value)
      } else {
        ElMessage.error('生成载体图像失败-Invalid response data:', res.data)
      }
    })
  }

  reader.readAsDataURL(file)
}

const toggleSidebar = () => {
  showSidebar.value = !showSidebar.value
}
//处理图像噪声模拟请求
const handleNoise = async () => {
  console.log(uploadFile.value.raw)
  console.log(input.value)
  const res = await axios.post('http://127.0.0.1:5000/noise', {
    file: uploadFile.value.raw,
  }, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
  //处理响应
  let data = res.data
  let splitData = data.split('####')
  resultMsg.value = splitData[0]
  console.info(resultMsg.value)
  let base64Image = splitData[1].split(';base64,').pop()
  let blob = await (await fetch(`data:image/jpeg;base64,${base64Image}`)).blob()
  resultImageUrl.value = URL.createObjectURL(blob)
  console.log(resultImageUrl.value)
}
</script>

<template>
  <main>
    <div style="display: flex;justify-content:flex-start;flex-flow:flex-start;">
      <!--左边竖放进度条 -->
      <!-- <el-progress v-if="currentTaskId" :percentage="progressPercentage" stroke-width="18" :status="progressStatus"> -->
      <!--</el-progress> -->
      <!-- 状态信息 -->
      <div v-if="currentTaskId">
        <p>任务ID: {{ currentTaskId }}</p>
        <p>状态: {{ taskStatus }}</p>
      </div>
      <div style="display: flex;justify-content:flex-start;flex-flow:column;">
  <!--      上半部分三个框，第一个框是上传框-->
        <div class="title">
          <el-span>原始图像生成</el-span>
          <el-span>载体图像生成</el-span>
          <el-span>加噪图像生成</el-span>
        </div>
        <div class="top">
          <el-row>
              <el-upload
                class="avatar-uploader"
                ref="uploadRef"
                action="https://run.mocky.io/v3/9d059bf9-4660-45f2-925d-ce80ad6c4d15"
                :show-file-list="false"
                :on-success="handleAvatarSuccess"
                :before-upload="beforeAvatarUpload"
                :auto-upload="false"
                :on-change="handleImgChange"
              >
                <img v-if="imageUrl" :src="imageUrl" class="avatar" alt="" />
                <el-icon v-else class="avatar-uploader-icon">
                  <Plus />
                </el-icon>
              </el-upload>
            </el-row>
            <el-row>
            <div class="receiveImg">
              <img v-if="resultImageUrl" class="receiveImg" :src="carrierImageUrl" alt="" />
            </div>
            </el-row>
          <el-row>
          <div class="receiveImg">
            <img v-if="resultImageUrl" class="receiveImg" :src="NoiseImageUrl" alt="" />
          </div>
          </el-row>
        </div>
        <div class="medium" style="display: flex;justify-content: flex-start;margin: 20px 0;">
          <el-row>
            <el-input v-model="input1" style="width:178px; margin-right:16px" placeholder="输入私钥"  />
          </el-row>
          <el-row>
            <el-input v-model="input2" style="width:178px; margin-right:16px" placeholder="输入公钥" />
          </el-row>
          <el-row>
            <el-button style="margin-top:3px" @click="toggleSidebar">噪声模拟</el-button>
          </el-row>
            <el-col>
              <div v-if="showSidebar" class="sidebar">
                <el-button @click="handleNoise">压缩</el-button>
                <el-button @click="handleNoise">裁剪</el-button>
                <el-button @click="handleNoise">加噪声</el-button>
                <el-button @click="handleNoise">光线变化</el-button>
              </div>
            </el-col>
        </div>
        <div class="bottom" style="display: flex; justify-content: space-evenly; margin-right:194px;margin-left:192px; flex-flow:column; width:186px;">
          <div class="receiveImg">
            <img v-if="resultImageUrl" class="receiveImg" :src="resultImageUrl" alt="" />
          </div>
          <div class="bottombuttons">
            <el-button style="width:100px" @click="handleSubmit(false)">获取载体图像</el-button>
            <el-button style="width:100px" @click="handleSubmit(true)">获取原始图像</el-button>
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
  color:aquamarine;
}
.title {
  display: flex;
  justify-content: flex-start;
  margin-bottom: 5px;
  margin-left:44px;
}
.title > *{
  margin-right: 104px;
}
main {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}
.avatar-uploader .avatar {
  width: 178px;
  height: 178px;
  display: block;
  margin-right: 16px;
}
.top{
  display:flex;
  justify-content: flex-start;
  align-items: center;
}
.top > :nth-child(2) {
  margin-right: 16px;
}
.top > :nth-child(1) {
  margin-right: 16px;
}
.medium > :nth-child(1), .medium > :nth-child(2) {
  margin-right: 16px ;
  width:178px;
  padding:0 ;
}
.medium .el-button{
  height:38px;
}
.bottombuttons{
  display:flex;
  justify-content: space-evenly;
  margin-top: 3px;
  margin-left: 39px;
  flex-direction: column;
}
.bottombuttons > *{
  margin:3px 0 0 0 !important;
}
</style>

<style>
.medium{
  height:40px;
}

.sidebar {
  display:flex;
  flex-flow: column;
  background-color: #333;
  width:80px;
  color: white;
  margin-left:10px;
  border-radius:5px;
}
.sidebar > * {
  margin:3px 0 0 0 !important;
}

.avatar-uploader .el-upload {
  border: 1px dashed var(--el-border-color);
  border-radius: 6px;
  cursor: pointer;
  //position: relative;
  //overflow: hidden;
  //transition: var(--el-transition-duration-fast);
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

</style>
