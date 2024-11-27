<template>
  <div class="top-bar">
    <img src="../../../topbaricon.png" class="icon" alt="Icon">
    <span class="title">燔石图像隐写平台</span>
  </div>
</template>

<script>
export default {
  name: 'TopBar'
}
</script>

<style scoped>
.top-bar {
  position: fixed; /* 固定定位 */
  top: 0;          /* 位于视图顶部 */
  left: 0;         /* 左侧对齐 */
  right: 0;        /* 右侧对齐，确保横跨整个宽度 */
  display: flex;
  align-items: center;
  padding: 10px 20px; /* 上下左右内边距 */
  background-color: #333;
  color: #fff;
  z-index: 1000;    /* 确保在其他元素之上 */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* 可选：添加阴影效果 */
}

.icon {
  width: 30px;
  height: 30px;
  margin-right: 10px;
}

.title {
  font-size: 20px;
  font-weight: bold; /* 可选：加粗标题 */
  color:aquamarine;
}
</style>
