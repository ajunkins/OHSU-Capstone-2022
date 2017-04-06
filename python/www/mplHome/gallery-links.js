/* global blueimp, $ */

$(function () {
    'use strict';

    // Initialize the Gallery as image carousel:
    blueimp.Gallery([
        {
        title: 'No Movement',
        href:  'img_arm_motions/No_Movement.png',
        type: 'image/png',
        thumbnail: 'img_arm_motions/No_Movement.png'
        },
        {
        title: 'Elbow Flexion',
        href:  'img_arm_motions/Elbow_Flexion.png',
        type: 'image/png',
        thumbnail: 'img_arm_motions/Elbow_Flexion.png'
        },
        {
        title: 'Elbow Extension',
        href:  'img_arm_motions/Elbow_Extension.png',
        type: 'image/png',
        thumbnail: 'img_arm_motions/Elbow_Extension.png'
        },
        {
        title: 'Wrist Rotate In (Pronate)',
        href:  'img_arm_motions/Wrist_Rotate_In.png',
        type: 'image/png',
        thumbnail: 'img_arm_motions/Wrist_Rotate_In.png'
        },
        {
        title: 'Wrist Rotate Out (Supinate)',
        href:  'img_arm_motions/Wrist_Rotate_Out.png',
        type: 'image/png',
        thumbnail: 'img_arm_motions/Wrist_Rotate_Out.png'
        },
        {
        title: 'Wrist Flex In',
        href:  'img_arm_motions/Wrist_Flexion.png',
        type: 'image/png',
        thumbnail: 'img_arm_motions/Wrist_Flexion.png'
        },
        {
        title: 'Wrist Extend Out',
        href:  'img_arm_motions/Wrist_Extension.png',
        type: 'image/png',
        thumbnail: 'img_arm_motions/Wrist_Extension.png'
        },
        {
        title: 'Hand Open',
        href:  'img_grasps/Hand_Open.png',
        type: 'image/png',
        thumbnail: 'img_grasps/Hand_Open.png'
        },
        {
        title: 'Spherical Grasp',
        href:  'img_grasps/Spherical_Grasp.png',
        type: 'image/png',
        thumbnail: 'img_grasps/Spherical_Grasp.png'
        },
        {
        title: 'Fine Pinch Grasp',
        href:  'img_grasps/Tip_Grasp.png',
        type: 'image/png',
        thumbnail: 'img_grasps/Tip_Grasp.png'
        },
        {
        title: 'Point Grasp',
        href:  'img_grasps/Trigger_Grasp.png',
        type: 'image/png',
        thumbnail: 'img_grasps/Trigger_Grasp.png'
        }
    ], {
        container: '#blueimp-image-carousel',
        carousel: true,
        onslide: function (index, slide) {
                // Callback function executed on slide change.
                switch(index) {
                case  0: sendCmd("Cls:No Movement"); break;
                case  1: sendCmd("Cls:Elbow Flexion"); break;
                case  2: sendCmd("Cls:Elbow Extension"); break;
                case  3: sendCmd("Cls:Wrist Rotate In"); break;
                case  4: sendCmd("Cls:Wrist Rotate Out"); break;
                case  5: sendCmd("Cls:Wrist Flex In"); break;
                case  6: sendCmd("Cls:Wrist Extend Out"); break;
                case  7: sendCmd("Cls:Hand Open"); break;
                case  8: sendCmd("Cls:Spherical Grasp"); break;
                case  9: sendCmd("Cls:Tip Grasp"); break;
                case 10: sendCmd("Cls:Point Grasp"); break;
                default: break;
                }        
        }
    });
}); // function
