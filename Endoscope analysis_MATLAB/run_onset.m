[D210810_m3_t1onsmean,D210810_m3_t1onsesd,D210810_m3_t1b4mean,D210810_m3_t1b4sd] = onsetalign(m3t110_mat,m3t110_time,140,593);
[D210810_m3_t2onsmean,D210810_m3_t2onsesd,D210810_m3_t2b4mean,D210810_m3_t2b4sd] = onsetalign(m3t210mat,m3t210cell,150,760);
[D210810_m3_t3onsmean,D210810_m3_t3onsesd,D210810_m3_t3b4mean,D210810_m3_t3b4sd] = onsetalign(m3t310_mat,m3t310cell,150,700);

[D210810_m5_t1onsmean,D210810_m5_t1onsesd,D210810_m5_t1b4mean,D210810_m5_t1b4sd] = onsetalign(m5t110_mat,m5t110_time,170,700);
[D210810_m5_t2onsmean,D210810_m5_t2onsesd,D210810_m5_t2b4mean,D210810_m5_t2b4sd] = onsetalign(m5t210_mat,m5t210_time,170,850);
[D210810_m5_t3onsmean,D210810_m5_t3onsesd,D210810_m5_t3b4mean,D210810_m5_t3b4sd] = onsetalign(m5t310_mat,m5t310_time,150,810);

[D210811_m3_t1onsmean,D210811_m3_t1onsesd,D210811_m3_t1b4mean,D210811_m3_t1b4sd] = onsetalign(m3t111mat,m3t111cell,170,700);
[D210811_m3_t2onsmean,D210811_m3_t2onsesd,D210811_m3_t2b4mean,D210811_m3_t2b4sd] = onsetalign(m3t211_mat,m3t211_time,180,610);
[D210811_m3_t3onsmean,D210811_m3_t3onsesd,D210811_m3_t3b4mean,D210811_m3_t3b4sd] = onsetalign(m3t311_mat,m3t311_time,300,730);

[D210811_m4_t1onsmean,D210811_m4_t1onsesd,D210811_m4_t1b4mean,D210811_m4_t1b4sd] = onsetalign(m4t111_mat,m4t111_time,200,780);
[D210811_m4_t2onsmean,D210811_m4_t2onsesd,D210811_m4_t2b4mean,D210811_m4_t2b4sd] = onsetalign(m4t211_mat,m4t211_time,200,540);
[D210811_m4_t3onsmean,D210811_m4_t3onsesd,D210811_m4_t3b4mean,D210811_m4_t3b4sd] = onsetalign(m4t311_mat,m4t311_time,200,490);

[D230811_m5_t1onsmean,D210811_m5_t1onsesd,D210811_m5_t1b4mean,D210811_m5_t1b4sd] = onsetalign(m5t111_mat,m5t111_time,190,540);
[D210811_m5_t2onsmean,D210811_m5_t2onsesd,D210811_m5_t2b4mean,D210811_m5_t2b4sd] = onsetalign(m5t211_mat,m5t211_time,160,540);
[D210811_m5_t3onsmean,D210811_m5_t3onsesd,D210811_m5_t3b4mean,D210811_m5_t3b4sd] = onsetalign(m5t311_mat,m5t311_time,120,400);

[D210812_m3_t1onsmean,D210812_m3_t1onsesd,D210812_m3_t1b4mean,D210812_m3_t1b4sd] = onsetalign(m3t112_mat,m3t112_time,1700,2060);
[D210812_m3_t2onsmean,D210812_m3_t2onsesd,D210812_m3_t2b4mean,D210812_m3_t2b4sd] = onsetalign(m3t212_mat,m3t212_time,170,620);
[D210812_m3_t3onsmean,D210812_m3_t3onsesd,D210812_m3_t3b4mean,D210812_m3_t3b4sd] = onsetalign(m3t312_mat,m3t312_time,160,500);

[D210812_m4_t1onsmean,D210812_m4_t1onsesd,D210812_m4_t1b4mean,D210812_m4_t1b4sd] = onsetalign(m4t112_mat,m4t112_time,150,450);
[D210812_m4_t3onsmean,D210812_m4_t3onsesd,D210812_m4_t3b4mean,D210812_m4_t3b4sd] = onsetalign(m4t312_mat,m4t312_time,250,580);

[D210812_m5_t1onsmean,D210812_m5_t1onsesd,D210812_m5_t1b4mean,D210812_m5_t1b4sd] = onsetalign(m5t112_mat,m5t112_time,110,400);
[D210812_m5_t3onsmean,D210812_m5_t3onsesd,D210812_m5_t3b4mean,D210812_m5_t3b4sd] = onsetalign(m5t312_mat,m5t312_time,80,570);



onsetplot2(D210810_m3_t1onsmean,D210810_m3_t1onsesd,D210810_m3_t1ind{1,1},3*ones(length(D210810_m3_t1ind{1,1}),1),D210810_m3_t1b4mean,D210810_m3_t1b4sd)
onsetplot2(D210810_m3_t2onsmean,D210810_m3_t2onsesd,D210810_m3_t2ind{1,1},3*ones(length(D210810_m3_t2ind{1,1}),1),D210810_m3_t2b4mean,D210810_m3_t2b4sd)
onsetplot2(D210810_m3_t3onsmean,D210810_m3_t3onsesd,D210810_m3_t3ind{1,1},3*ones(length(D210810_m3_t3ind{1,1}),1),D210810_m3_t3b4mean,D210810_m3_t3b4sd)

onsetplot2(D210810_m5_t1onsmean,D210810_m5_t1onsesd,D210810_m5_t1ind{1,1},3*ones(length(D210810_m5_t1ind{1,1}),1),D210810_m5_t1b4mean,D210810_m5_t1b4sd)
onsetplot2(D210810_m5_t2onsmean,D210810_m5_t2onsesd,D210810_m5_t2ind{1,1},3*ones(length(D210810_m5_t2ind{1,1}),1),D210810_m5_t2b4mean,D210810_m5_t2b4sd)
onsetplot2(D210810_m5_t3onsmean,D210810_m5_t3onsesd,D210810_m5_t3ind{1,1},3*ones(length(D210810_m5_t3ind{1,1}),1),D210810_m5_t3b4mean,D210810_m5_t3b4sd)

onsetplot2(D210811_m3_t1onsmean,D210811_m3_t1onsesd,D210811_m3_t1ind{1,1},3*ones(length(D210811_m3_t1ind{1,1}),1),D210811_m3_t1b4mean,D210811_m3_t1b4sd)
onsetplot2(D210811_m3_t2onsmean,D210811_m3_t2onsesd,D210811_m3_t2ind{1,1},3*ones(length(D210811_m3_t2ind{1,1}),1),D210811_m3_t2b4mean,D210811_m3_t2b4sd)
onsetplot2(D210811_m3_t3onsmean,D210811_m3_t3onsesd,D210811_m3_t3ind{1,1},3*ones(length(D210811_m3_t3ind{1,1}),1),D210811_m3_t3b4mean,D210811_m3_t3b4sd)

onsetplot2(D210811_m4_t1onsmean,D210811_m4_t1onsesd,D210811_m4_t1ind{1,1},3*ones(length(D210811_m4_t1ind{1,1}),1),D210811_m4_t1b4mean,D210811_m4_t1b4sd)
onsetplot2(D210811_m4_t2onsmean,D210811_m4_t2onsesd,D210811_m4_t2ind{1,1},3*ones(length(D210811_m4_t2ind{1,1}),1),D210811_m4_t2b4mean,D210811_m4_t2b4sd)
onsetplot2(D210811_m4_t3onsmean,D210811_m4_t3onsesd,D210811_m4_t3ind{1,1},3*ones(length(D210811_m4_t3ind{1,1}),1),D210811_m4_t3b4mean,D210811_m4_t3b4sd)

onsetplot2(D210811_m5_t1onsmean,D210811_m5_t1onsesd,D210811_m5_t1ind{1,1},3*ones(length(D210811_m5_t1ind{1,1}),1),D210811_m5_t1b4mean,D210811_m5_t1b4sd)
onsetplot2(D210811_m5_t2onsmean,D210811_m5_t2onsesd,D210811_m5_t2ind{1,1},3*ones(length(D210811_m5_t2ind{1,1}),1),D210811_m5_t2b4mean,D210811_m5_t2b4sd)
onsetplot2(D210811_m5_t3onsmean,D210811_m5_t3onsesd,D210811_m5_t3ind{1,1},3*ones(length(D210811_m5_t3ind{1,1}),1),D210811_m5_t3b4mean,D210811_m5_t3b4sd)

onsetplot2(D210812_m3_t1onsmean,D210812_m3_t1onsesd,D210812_m3_t1ind{1,1},3*ones(length(D210812_m3_t1ind{1,1}),1),D210812_m3_t1b4mean,D210812_m3_t1b4sd)
onsetplot2(D210812_m3_t2onsmean,D210812_m3_t2onsesd,D210812_m3_t2ind{1,1},3*ones(length(D210812_m3_t2ind{1,1}),1),D210812_m3_t2b4mean,D210812_m3_t2b4sd)
onsetplot2(D210812_m3_t3onsmean,D210812_m3_t3onsesd,D210812_m3_t3ind{1,1},3*ones(length(D210812_m3_t3ind{1,1}),1),D210812_m3_t3b4mean,D210812_m3_t3b4sd)

onsetplot2(D210812_m4_t1onsmean,D210812_m4_t1onsesd,D210812_m4_t1ind{1,1},3*ones(length(D210812_m4_t1ind{1,1}),1),D210812_m4_t1b4mean,D210812_m4_t1b4sd)
onsetplot2(D210812_m4_t3onsmean,D210812_m4_t3onsesd,D210812_m4_t3ind{1,1},3*ones(length(D210812_m4_t3ind{1,1}),1),D210812_m4_t3b4mean,D210812_m4_t3b4sd)

onsetplot2(D210812_m5_t1onsmean,D210812_m5_t1onsesd,D210812_m5_t1ind{1,1},3*ones(length(D210812_m5_t1ind{1,1}),1),D210812_m5_t1b4mean,D210812_m5_t1b4sd)
onsetplot2(D210812_m5_t3onsmean,D210812_m5_t3onsesd,D210812_m5_t3ind{1,1},3*ones(length(D210812_m5_t3ind{1,1}),1),D210812_m5_t3b4mean,D210812_m5_t3b4sd)

onsetplot2(D210810_m3_t1onsmean,D210810_m3_t1onsesd,D210810_m3_t1ind{2,1},3*ones(length(D210810_m3_t1ind{2,1}),1),D210810_m3_t1b4mean,D210810_m3_t1b4sd)
onsetplot2(D210810_m3_t2onsmean,D210810_m3_t2onsesd,D210810_m3_t2ind{2,1},3*ones(length(D210810_m3_t2ind{2,1}),1),D210810_m3_t2b4mean,D210810_m3_t2b4sd)
onsetplot2(D210810_m3_t3onsmean,D210810_m3_t3onsesd,D210810_m3_t3ind{2,1},3*ones(length(D210810_m3_t3ind{2,1}),1),D210810_m3_t3b4mean,D210810_m3_t3b4sd)

onsetplot2(D210810_m5_t1onsmean,D210810_m5_t1onsesd,D210810_m5_t1ind{2,1},3*ones(length(D210810_m5_t1ind{2,1}),1),D210810_m5_t1b4mean,D210810_m5_t1b4sd)
onsetplot2(D210810_m5_t2onsmean,D210810_m5_t2onsesd,D210810_m5_t2ind{2,1},3*ones(length(D210810_m5_t2ind{2,1}),1),D210810_m5_t2b4mean,D210810_m5_t2b4sd)
onsetplot2(D210810_m5_t3onsmean,D210810_m5_t3onsesd,D210810_m5_t3ind{2,1},3*ones(length(D210810_m5_t3ind{2,1}),1),D210810_m5_t3b4mean,D210810_m5_t3b4sd)

onsetplot2(D210811_m3_t1onsmean,D210811_m3_t1onsesd,D210811_m3_t1ind{2,1},3*ones(length(D210811_m3_t1ind{2,1}),1),D210811_m3_t1b4mean,D210811_m3_t1b4sd)
onsetplot2(D210811_m3_t2onsmean,D210811_m3_t2onsesd,D210811_m3_t2ind{2,1},3*ones(length(D210811_m3_t2ind{2,1}),1),D210811_m3_t2b4mean,D210811_m3_t2b4sd)
onsetplot2(D210811_m3_t3onsmean,D210811_m3_t3onsesd,D210811_m3_t3ind{2,1},3*ones(length(D210811_m3_t3ind{2,1}),1),D210811_m3_t3b4mean,D210811_m3_t3b4sd)

onsetplot2(D210811_m4_t1onsmean,D210811_m4_t1onsesd,D210811_m4_t1ind{2,1},3*ones(length(D210811_m4_t1ind{2,1}),1),D210811_m4_t1b4mean,D210811_m4_t1b4sd)
onsetplot2(D210811_m4_t2onsmean,D210811_m4_t2onsesd,D210811_m4_t2ind{2,1},3*ones(length(D210811_m4_t2ind{2,1}),1),D210811_m4_t2b4mean,D210811_m4_t2b4sd)
onsetplot2(D210811_m4_t3onsmean,D210811_m4_t3onsesd,D210811_m4_t3ind{2,1},3*ones(length(D210811_m4_t3ind{2,1}),1),D210811_m4_t3b4mean,D210811_m4_t3b4sd)

onsetplot2(D210811_m5_t1onsmean,D210811_m5_t1onsesd,D210811_m5_t1ind{2,1},3*ones(length(D210811_m5_t1ind{2,1}),1),D210811_m5_t1b4mean,D210811_m5_t1b4sd)
onsetplot2(D210811_m5_t2onsmean,D210811_m5_t2onsesd,D210811_m5_t2ind{2,1},3*ones(length(D210811_m5_t2ind{2,1}),1),D210811_m5_t2b4mean,D210811_m5_t2b4sd)
onsetplot2(D210811_m5_t3onsmean,D210811_m5_t3onsesd,D210811_m5_t3ind{2,1},3*ones(length(D210811_m5_t3ind{2,1}),1),D210811_m5_t3b4mean,D210811_m5_t3b4sd)

onsetplot2(D210812_m3_t1onsmean,D210812_m3_t1onsesd,D210812_m3_t1ind{2,1},3*ones(length(D210812_m3_t1ind{2,1}),1),D210812_m3_t1b4mean,D210812_m3_t1b4sd)
onsetplot2(D210812_m3_t2onsmean,D210812_m3_t2onsesd,D210812_m3_t2ind{2,1},3*ones(length(D210812_m3_t2ind{2,1}),1),D210812_m3_t2b4mean,D210812_m3_t2b4sd)
onsetplot2(D210812_m3_t3onsmean,D210812_m3_t3onsesd,D210812_m3_t3ind{2,1},3*ones(length(D210812_m3_t3ind{2,1}),1),D210812_m3_t3b4mean,D210812_m3_t3b4sd)

onsetplot2(D210812_m4_t1onsmean,D210812_m4_t1onsesd,D210812_m4_t1ind{2,1},3*ones(length(D210812_m4_t1ind{2,1}),1),D210812_m4_t1b4mean,D210812_m4_t1b4sd)
onsetplot2(D210812_m4_t3onsmean,D210812_m4_t3onsesd,D210812_m4_t3ind{2,1},3*ones(length(D210812_m4_t3ind{2,1}),1),D210812_m4_t3b4mean,D210812_m4_t3b4sd)

onsetplot2(D210812_m5_t1onsmean,D210812_m5_t1onsesd,D210812_m5_t1ind{2,1},3*ones(length(D210812_m5_t1ind{2,1}),1),D210812_m5_t1b4mean,D210812_m5_t1b4sd)
onsetplot2(D210812_m5_t3onsmean,D210812_m5_t3onsesd,D210812_m5_t3ind{2,1},3*ones(length(D210812_m5_t3ind{2,1}),1),D210812_m5_t3b4mean,D210812_m5_t3b4sd)

onsetplot2(D210810_m3_t1onsmean,D210810_m3_t1onsesd,D210810_m3_t1ind{3,1},3*ones(length(D210810_m3_t1ind{3,1}),1),D210810_m3_t1b4mean,D210810_m3_t1b4sd)
onsetplot2(D210810_m3_t2onsmean,D210810_m3_t2onsesd,D210810_m3_t2ind{3,1},3*ones(length(D210810_m3_t2ind{3,1}),1),D210810_m3_t2b4mean,D210810_m3_t2b4sd)
onsetplot2(D210810_m3_t3onsmean,D210810_m3_t3onsesd,D210810_m3_t3ind{3,1},3*ones(length(D210810_m3_t3ind{3,1}),1),D210810_m3_t3b4mean,D210810_m3_t3b4sd)

onsetplot2(D210810_m5_t1onsmean,D210810_m5_t1onsesd,D210810_m5_t1ind{3,1},3*ones(length(D210810_m5_t1ind{3,1}),1),D210810_m5_t1b4mean,D210810_m5_t1b4sd)
onsetplot2(D210810_m5_t2onsmean,D210810_m5_t2onsesd,D210810_m5_t2ind{3,1},3*ones(length(D210810_m5_t2ind{3,1}),1),D210810_m5_t2b4mean,D210810_m5_t2b4sd)
onsetplot2(D210810_m5_t3onsmean,D210810_m5_t3onsesd,D210810_m5_t3ind{3,1},3*ones(length(D210810_m5_t3ind{3,1}),1),D210810_m5_t3b4mean,D210810_m5_t3b4sd)

onsetplot2(D210811_m3_t1onsmean,D210811_m3_t1onsesd,D210811_m3_t1ind{3,1},3*ones(length(D210811_m3_t1ind{3,1}),1),D210811_m3_t1b4mean,D210811_m3_t1b4sd)
onsetplot2(D210811_m3_t2onsmean,D210811_m3_t2onsesd,D210811_m3_t2ind{3,1},3*ones(length(D210811_m3_t2ind{3,1}),1),D210811_m3_t2b4mean,D210811_m3_t2b4sd)
onsetplot2(D210811_m3_t3onsmean,D210811_m3_t3onsesd,D210811_m3_t3ind{3,1},3*ones(length(D210811_m3_t3ind{3,1}),1),D210811_m3_t3b4mean,D210811_m3_t3b4sd)

onsetplot2(D210811_m4_t1onsmean,D210811_m4_t1onsesd,D210811_m4_t1ind{3,1},3*ones(length(D210811_m4_t1ind{3,1}),1),D210811_m4_t1b4mean,D210811_m4_t1b4sd)
onsetplot2(D210811_m4_t2onsmean,D210811_m4_t2onsesd,D210811_m4_t2ind{3,1},3*ones(length(D210811_m4_t2ind{3,1}),1),D210811_m4_t2b4mean,D210811_m4_t2b4sd)
onsetplot2(D210811_m4_t3onsmean,D210811_m4_t3onsesd,D210811_m4_t3ind{3,1},3*ones(length(D210811_m4_t3ind{3,1}),1),D210811_m4_t3b4mean,D210811_m4_t3b4sd)

onsetplot2(D210811_m5_t1onsmean,D210811_m5_t1onsesd,D210811_m5_t1ind{3,1},3*ones(length(D210811_m5_t1ind{3,1}),1),D210811_m5_t1b4mean,D210811_m5_t1b4sd)
onsetplot2(D210811_m5_t2onsmean,D210811_m5_t2onsesd,D210811_m5_t2ind{3,1},3*ones(length(D210811_m5_t2ind{3,1}),1),D210811_m5_t2b4mean,D210811_m5_t2b4sd)
onsetplot2(D210811_m5_t3onsmean,D210811_m5_t3onsesd,D210811_m5_t3ind{3,1},3*ones(length(D210811_m5_t3ind{3,1}),1),D210811_m5_t3b4mean,D210811_m5_t3b4sd)

onsetplot2(D210812_m3_t1onsmean,D210812_m3_t1onsesd,D210812_m3_t1ind{3,1},3*ones(length(D210812_m3_t1ind{3,1}),1),D210812_m3_t1b4mean,D210812_m3_t1b4sd)
onsetplot2(D210812_m3_t2onsmean,D210812_m3_t2onsesd,D210812_m3_t2ind{3,1},3*ones(length(D210812_m3_t2ind{3,1}),1),D210812_m3_t2b4mean,D210812_m3_t2b4sd)
onsetplot2(D210812_m3_t3onsmean,D210812_m3_t3onsesd,D210812_m3_t3ind{3,1},3*ones(length(D210812_m3_t3ind{3,1}),1),D210812_m3_t3b4mean,D210812_m3_t3b4sd)

onsetplot2(D210812_m4_t1onsmean,D210812_m4_t1onsesd,D210812_m4_t1ind{3,1},3*ones(length(D210812_m4_t1ind{3,1}),1),D210812_m4_t1b4mean,D210812_m4_t1b4sd)
onsetplot2(D210812_m4_t3onsmean,D210812_m4_t3onsesd,D210812_m4_t3ind{3,1},3*ones(length(D210812_m4_t3ind{3,1}),1),D210812_m4_t3b4mean,D210812_m4_t3b4sd)

onsetplot2(D210812_m5_t1onsmean,D210812_m5_t1onsesd,D210812_m5_t1ind{3,1},3*ones(length(D210812_m5_t1ind{3,1}),1),D210812_m5_t1b4mean,D210812_m5_t1b4sd)
onsetplot2(D210812_m5_t3onsmean,D210812_m5_t3onsesd,D210812_m5_t3ind{3,1},3*ones(length(D210812_m5_t3ind{3,1}),1),D210812_m5_t3b4mean,D210812_m5_t3b4sd)