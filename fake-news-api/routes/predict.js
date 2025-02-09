const express=require('express')
const router=express.Router()

const {predictNews}=require('../controllers/predictController.js')


router.post('/',predictNews);

module.exports=router;