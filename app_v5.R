#setwd("C:/Users/cbatti545@cable.comcast.com/OneDrive - Comcast/Documents_old/football_apps/gm_app")


#.libPaths("C:/Users/cbatti545@cable.comcast.com/OneDrive - Comcast/Documents_old/R/R-4.1.3/library")

#_LIBS="C:/Users/cbatti545@cable.comcast.com/OneDrive - Comcast/Documents_old/R/R-4.1.3/library"

#install.packages("r")
library(scrypt)
library(shinymanager)
library(tidyverse)
library(data.table)
library(lubridate)
library(shinydashboard)
library(readr)
library(stats)
#library(tidymodels)
library(ggplot2)
library(ggpmisc)
library(readxl)
library(data.table)
library(rjson)
library(pacman)
#library(rlang)
library(gt)
library(gtExtras)
library(stringi)
library(fuzzyjoin)
library(factoextra)
library(rsample)
library(plotly)
library(shiny)
library(bootstrap)
library(bslib)
library(flexdashboard)
library(ggalluvial)
#install.packages("scrypt")



inactivity <- "function idleTimer() {
var t = setTimeout(logout, 120000);
window.onmousemove = resetTimer; // catches mouse movements
window.onmousedown = resetTimer; // catches mouse movements
window.onclick = resetTimer;     // catches mouse clicks
window.onscroll = resetTimer;    // catches scrolling
window.onkeypress = resetTimer;  //catches keyboard actions

function logout() {
window.close();  //close the window
}

function resetTimer() {
clearTimeout(t);
t = setTimeout(logout, 120000);  // time is in milliseconds (1000 is 1 second)
}
}
idleTimer();"

credentials <- data.frame(
  user = c("mifflin_staff", "fanny", "victor", "benoit"),
  password = "beatwilson!",
  is_hashed_password = FALSE,
  # comment = c("alsace", "auvergne", "bretagne"), %>% 
  stringsAsFactors = FALSE
)


pbp <- fread(file="sample_data_gm_football.csv")%>% 
  rename(playcall_='Play call') %>% 
  mutate(playcall=ifelse(play_type=="pass",paste0(play_type,"_",pass_length,"_",pass_location),paste0(play_type,"_",run_gap,"_",run_location))) %>% 
  filter(playcall != "pass_NA_NA", qtr !=5, !is.na(down))


pbp_grouped <- pbp %>% 
  group_by(Formation,playcall) %>% 
  summarise(avg_yds_gained=round(mean(yds_gained),1),
            count_=n()) %>% 
  mutate(Formation=as.character(Formation))

pbp_grouped$Formation <- factor(pbp_grouped$Formation,
                                         levels = c("1", "2", "3", "4","5", "6", "7","8","9","10"))


pbp_grouped_run_plays <- pbp %>%
  filter(play_type=='run') %>% 
  group_by(down,playcall) %>% 
  summarise(avg_yds_gained=round(mean(yds_gained),1)) %>% 
  filter(playcall != 'run_NA_NA') 

pbp_grouped_run_plays$playcall <- factor(pbp_grouped_run_plays$playcall,
                                         levels = c("run_end_right", "run_tackle_right", "run_guard_right", "run_NA_middle","run_guard_left", "run_tackle_left", "run_end_left"))


pbp_grouped_pass_plays <- pbp %>%
  filter(play_type=='pass') %>% 
  group_by(down,pass_length,pass_location) %>% 
  summarise(comp_pct=round(sum(yds_gained==0)/n(),2))

pbp_grouped_pass_plays$pass_length <- factor(pbp_grouped_pass_plays$pass_length,
                                         levels = c("short", "deep"))


pbp_grouped_playcall <- pbp %>% 
  mutate(first_down_gained=ifelse(yds_gained>distance,1,0))%>%
  group_by(playcall,down) %>% 
  summarise(avg_yds_togo=round(mean(distance),1),
            avg_yds_gained=round(mean(yds_gained),1),
            conversion_pct=round(((sum(first_down_gained)/n())*100),1),
            )


pbp_grouped_exec <- pbp %>% 
  mutate(first_down_gained=ifelse(yds_gained>distance,1,0))%>% 
  group_by(down) %>% 
  summarise(avg_yds_togo=round(mean(distance),1),
    avg_yds_gained=round(mean(yds_gained),1),
            conversion_pct=round(((sum(first_down_gained)/n())*100),1)) %>% 
  rename('Avg Yds To Go'=avg_yds_togo,'Avg Yds Gained'=avg_yds_gained,'Conversion %'=conversion_pct) 

pbp_grouped_exec_piv <- pbp %>% 
  mutate(first_down_gained=ifelse(yds_gained>distance,1,0))%>% 
  group_by(down) %>% 
  summarise(avg_yds_togo=round(mean(distance),1),
            avg_yds_gained=round(mean(yds_gained),1),
            conversion_pct=round(((sum(first_down_gained)/n())*100),1)) %>% 
  rename('Avg Yds To Go'=avg_yds_togo,'Avg Yds Gained'=avg_yds_gained,'Conversion %'=conversion_pct) %>% 
  pivot_longer( cols = -down)

pbp_grouped_qtr <- pbp %>% 
  mutate(first_down_gained=ifelse(yds_gained>distance,1,0))%>% 
  group_by(qtr,down) %>% 
  summarise(avg_yds_togo=round(mean(distance),1),
            avg_yds_gained=round(mean(yds_gained),1),
            conversion_pct=round(((sum(first_down_gained)/n())*100),1)) %>% 
  rename('Avg Yds To Go'=avg_yds_togo,'Avg Yds Gained'=avg_yds_gained,'Conversion %'=conversion_pct)

pbp_grouped_qtr$line_size <- ifelse(pbp_grouped_qtr$down %in% c(3, 4), 3, 1.5)

formations_ <- pbp %>% 
  dplyr::pull(Formation) %>% 
  unique() %>% 
  sort()

playcalls_ <- pbp %>% 
  dplyr::pull(playcall) %>% 
  unique() %>% 
  sort()

playtype_ <- pbp %>% 
  dplyr::pull(play_type) %>% 
  unique() %>% 
  sort()

ui <- 
  #secure_app(head_auth = tags$script(inactivity),
  fluidPage(
  page_navbar(
    theme= bs_theme(
      bg = "#3b3b3b", fg = "#c5b358", primary = "#c5b358", secondary = "white",
      base_font = font_google("Ubuntu"),
      code_font = font_google("Space Mono")
    ),
    nav_panel("Offense - Execution",
              titlePanel("Offense - Execution"),
              layout_columns(card(card(card_title("1st Down")),
                                  card(card_header(class = "bg-dark","Summary"),
                                                                    gt_output("first_to_go")),
                                  card(card_header(class = "bg-dark","By Playcall"),
                                       navset_tab(nav_panel(title = "Avg Yds",plotOutput("first_gained_lollipop")),
                                                  nav_panel(title="Conv. %",plotOutput("first_conv_lollipop"))))),
                             card(card(card_title("2nd Down")),
                                  card(card_header(class = "bg-dark","Summary"),
                                       gt_output("sec_to_go")),
                                  card(card_header(class = "bg-dark","By Playcall"),
                                       navset_tab(nav_panel(title = "Avg Yds",plotOutput("second_gained_lollipop")),
                                                  nav_panel(title="Conv. %",plotOutput("second_conv_lollipop"))))),
                             card(card(card_title("3rd Down")),
                                  card(card_header(class = "bg-dark","Summary"),
                                       gt_output("third_to_go")),
                                  card(card_header(class = "bg-dark","By Playcall"),
                                       navset_tab(nav_panel(title = "Avg Yds",plotOutput("third_gained_lollipop")),
                                                  nav_panel(title="Conv. %",plotOutput("third_conv_lollipop"))))),
                             card(card(card_title("4th Down")),
                                  card(card_header(class = "bg-dark","Summary"),
                                       gt_output("fourth_to_go")),
                                  card(card_header(class = "bg-dark","By Playcall"),
                                       navset_tab(nav_panel(title = "Avg Yds",plotOutput("fourth_gained_lollipop")),
                                                  nav_panel(title="Conv. %",plotOutput("fourth_conv_lollipop"))))),
                             card(card_header(class = "bg-dark","Playcall x Formation Execution"),
                                  plotOutput("heatmap")),
                             card(card_header(class = "bg-dark","Run Execution"),
                                  plotOutput("run_flow_chart")
                                  ),
                             card(card_header(class = "bg-dark","Pass Execution"),
                                  plotOutput("heatmap_pass")
                             ),
                             card(navset_tab(
                               nav_panel(title = "Avg Yards To Go by Qtr", plotOutput("togo_byqtr")),
                               nav_panel(title = "Avg Yards Gained by Qtr", plotOutput("gained_byqtr")),
                               nav_panel(title = "Conversion % by Qtr", plotOutput("conv_byqtr"))
                             )),
                             col_widths = c(3,3,3,3,12,12,12,12)
                             )
              ),
    nav_panel("Offense - Filters", 
              titlePanel("Offense - Filters"),
              layout_sidebar(title="sidebar____",
                sidebar= sidebar("SCROLL TO BOTTOM TO COLLAPSE",
                  checkboxGroupInput("formationsinput", label = "Formation(s):", choices = formations_,  selected=formations_[1:10]),
                  uiOutput("play_type_choices"),
                  uiOutput("play_call_choices_")
                ),
                  plotOutput("histplot"),
                  plotlyOutput("distanceVyds_gained")
                )
              ), 
    nav_panel("Offense - Tendencies",
              titlePanel("Offense - Tendencies"),
              layout_columns(
              card(card(card_title("1st Down")),card(card_header(class = "bg-dark","Playcall"),plotOutput("histplot_first")),card(card_header(class = "bg-dark","Formation"),plotOutput("histplot_first_form"))),
              card(card(card_title("2nd Down")),card(card_header(class = "bg-dark","Playcall"),plotOutput("histplot_second")),card(card_header(class = "bg-dark","Formation"),plotOutput("histplot_second_form"))),
              card(card(card_title("3rd Down")),card(card_header(class = "bg-dark","Playcall"),plotOutput("histplot_third")),card(card_header(class = "bg-dark","Formation"),plotOutput("histplot_third_form"))),
              card(card(card_title("4th Down")),card(card_header(class = "bg-dark","Playcall"),plotOutput("histplot_fourth")),card(card_header(class = "bg-dark","Formation"),plotOutput("histplot_fourth_form"))),
              card(card_header(class="bg-dark","Formation x Playcall Volume"),plotOutput("heatmap_tend",width = "100%")),
              col_widths = c(3,3,3,3,12))
              ), 
    nav_panel("All Plays", 
              tableOutput("raw_table")
              ), 
    title = "Governor Mifflin Football Analytics", 
    id = "page", 

)
)
#)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

server <- function(input, output, session) {
  
  ##################################login 
  result_auth <- secure_server(check_credentials = check_credentials(credentials))
  
  output$res_auth <- renderPrint({
    reactiveValuesToList(result_auth)
  })
  
  output$user_table <- renderTable({
    # use req to only render results when credentials()$user_auth is TRUE
    req(credentials()$user_auth)
    credentials()$info
  })
######################################
  ###ui outputs
  
  output$play_type_choices <-  renderUI({
    
    checkboxGroupInput("play_typeinput",
                       label = "Play Type(s):", 
                       choices = playtype_,  
                       selected=playtype_[1:2])
    
    
    
  })
  outputOptions(output, "play_type_choices", suspendWhenHidden = FALSE)
  
  output$play_call_choices_ <-  renderUI({
    
    checkboxGroupInput("play_callsinput",
                       label = "Play Call(s):", 
                       choices = filtered_playcall(),  
                       selected=filtered_playcall())
    
    
    
  })
  outputOptions(output, "play_call_choices_", suspendWhenHidden = FALSE)
  
  
####################################  
  
  first_down_plays <- pbp %>% 
    filter(down==1)
  
  second_down_plays <- pbp %>% 
    filter(down==2)

  third_down_plays <- pbp %>% 
    filter(down==3)
  
  
  fourth_down_plays <- pbp %>% 
    filter(down==4)
    
###########reactive
  
  filtered_playcall <- reactive(
    filtered_pbp_og() %>%
      filter(play_type %in% input$play_typeinput) %>% 
      pull(playcall) %>% 
      unique()
    
  )
  
  filtered_pbp_og <- reactive({
    pbp %>%
      mutate(color_code=ifelse(play_type=="pass","#c5b358","#550000"),
             first_down_gained=ifelse(yds_gained>distance,"1st Down","Other")) %>% 
      filter(Formation %in% input$formationsinput,
             play_type %in% input$play_typeinput)
    
  })
  
  filtered_pbp <- reactive(
    filtered_pbp_og() %>% 
      filter(playcall %in% input$play_callsinput)
  )
  
  
  
  filtered_pbp_grouped <- reactive(
    filtered_pbp() %>% 
    group_by(Formation,playcall) %>% 
    summarise(avg_yds_gained=mean(yds_gained),
              count_=n()))
##########outputs  
  
###############all plays table  
  output$raw_table <- renderTable({
    pbp
  }, width="100%", align = "c")
  
  
################filtered tab  
  output$distanceVyds_gained <- renderPlotly({
    
    ggplotly(ggplot(filtered_pbp(),aes(x=distance,y=yds_gained, color=play_type))+
      geom_point(alpha=.5,size=2, aes(shape=filtered_pbp()$play_type, fill=filtered_pbp()$play_type))+
      theme_linedraw(base_size = 20)+
      scale_fill_manual("Play Type", values=c("#550000","#c5b358"), labels=filtered_pbp()$play_type ,guide="none")+
      scale_color_manual(values=c("#550000","#c5b358"),guide="none")+  
      scale_shape_manual(values=c(23,21),guide="none")+  
      guides(fill= guide_legend(title = "Play Type"))+
      theme(legend.position = "bottom",
              legend.background = element_rect(fill = "white"),
            legend.key.size = unit(1, 'cm'), #change legend key size
            legend.key.height = unit(1, 'cm'), #change legend key height
            legend.key.width = unit(1, 'cm'),
            legend.title = element_text(size=14), legend.text =element_text(size=12))+
        labs(x ="Yds To Go", y="Yds Gained")+
        geom_abline(intercept = 0, slope = 1)
    )
  })
  
  output$histplot <- renderPlot({
    ggplot(filtered_pbp(),aes(x=down,fill=play_type,color=play_type))+
      geom_bar()+
      theme_linedraw(base_size = 20)+
      scale_color_manual(values=c("#550000","#c5b358"),guide="none")+
      scale_fill_manual(values=c("#550000","#c5b358"))+
      guides(fill = guide_legend(title = "Play Type"))+
      theme(legend.position = c(0.95, 0.85),
            legend.background = element_rect(fill = "white"),
            panel.grid.minor = element_blank(),
            panel.grid.major.x = element_blank())+
      labs(x="Down",y="Count")
    # + 
    #   scale_y_continuous(breaks = breaks_pretty())
  })
  
  
#############################exectution tab
  output$heatmap <- renderPlot({
    ggplot(pbp_grouped, aes(x = Formation, y = playcall, fill = avg_yds_gained)) +
      geom_tile(color = "black", linewidth=.75) +
      geom_text(aes(label = avg_yds_gained),  size =7) +
      scale_fill_gradient(name="Avg Yards Gained:", low = "white", high = "darkgreen")+
      theme_linedraw(base_size = 20)+
      ylab("Playcall")+
      xlab("Formation")+
      theme(panel.background = element_rect("gray"),
            panel.grid.major = element_blank(),
            legend.position="bottom",  legend.text = element_text(hjust = 0.5, vjust = 1, angle = 90))
  })
  
  output$heatmap_pass <- renderPlot({
    ggplot(pbp_grouped_pass_plays, aes(x = pass_location, y = pass_length, fill = comp_pct)) +
      geom_tile(color = "black", linewidth=.75) +
      geom_text(aes(label=comp_pct),  size =7,check_overlap = TRUE) +
      scale_fill_gradient(name="Comp Pct:", low = "white", high = "darkgreen")+
      theme_linedraw(base_size = 20)+
      ylab("Length")+
      xlab("Location")+
      theme(panel.background = element_rect("gray"),
            panel.grid.major = element_blank(),
            legend.position="bottom",  legend.text = element_text(hjust = 1, vjust = 1, angle = 90))+ 
      facet_wrap(vars(down), nrow = 1)
  })
  
  output$first_to_go <- render_gt({
    
    first_stats <-  pbp_grouped_exec_piv  %>% 
      filter(down==1) %>% 
      dplyr::select(-down) %>% 
      gt() %>% 
      tab_options(table.width=pct(80)) %>%
      cols_align(align='center', columns = everything()) %>%
      gt_theme_538() %>% 
      gt_theme_dark() %>% 
      tab_options(table.font.size = 22) %>% 
      data_color(
        columns = everything(),
        palette = "#550000"
      ) %>% 
      tab_style(style=cell_fill(color="#c5b358"),locations = cells_body(rows= 3 ))  
    
    
  })
  
  output$sec_to_go <- render_gt({
    pbp_grouped_exec_piv  %>% 
      filter(down==2) %>% 
      dplyr::select(-down) %>% 
      gt() %>% 
      tab_options(table.width=pct(80)) %>%
      cols_align(align='center', columns = everything()) %>%
      gt_theme_538() %>% 
      gt_theme_dark() %>% 
      tab_options(table.font.size = 22) %>% 
      data_color(
        columns = everything(),
        palette = "#550000"
      )%>% 
      tab_style(style=cell_fill(color="#c5b358"),locations = cells_body(rows= 3 ))
  })
  
  output$third_to_go <- render_gt({
    pbp_grouped_exec_piv  %>% 
      filter(down==3) %>% 
      dplyr::select(-down) %>% 
      gt() %>% 
      tab_options(table.width=pct(80)) %>%
      cols_align(align='center', columns = everything()) %>%
      gt_theme_538() %>% 
      gt_theme_dark() %>% 
      tab_options(table.font.size = 22) %>% 
      data_color(
        columns = everything(),
        palette = "#550000"
      )%>% 
      tab_style(style=cell_fill(color="#c5b358"),locations = cells_body(rows= 3 ))
  })
  
  output$fourth_to_go <- render_gt({
    pbp_grouped_exec_piv  %>% 
      filter(down==4) %>% 
      dplyr::select(-down) %>% 
      gt() %>% 
      tab_options(table.width=pct(80)) %>%
      cols_align(align='center', columns = everything()) %>%
      gt_theme_538() %>% 
      gt_theme_dark() %>% 
      tab_options(table.font.size = 22) %>% 
      data_color(
        columns = everything(),
        palette = "#550000"
      ) %>% 
      tab_style(style=cell_fill(color="#c5b358"),locations = cells_body(rows= 3 ))
  })
  
  output$conv_byqtr <- renderPlot({
  ggplot(pbp_grouped_qtr, aes(x=qtr,y=!!as.name('Conversion %'), group = down))+
      geom_line(aes(size=line_size), color="black")+
      geom_line(aes(size=1.5, color=factor(down)))+
      labs(color="Down")+
      xlab("Quarter")+
      theme_linedraw(base_size = 16)+
      theme(panel.grid.minor = element_blank())+
      scale_size_identity()
  })
  
  output$togo_byqtr <- renderPlot({
    
    
    ggplot(pbp_grouped_qtr, aes(x=qtr,y=!!as.name('Avg Yds To Go'), group = down))+
      geom_line(aes(size=line_size), color="black")+
      geom_line(aes(size=1.5, color=factor(down)))+
      labs(color="Down")+
      xlab("Quarter")+
      theme_linedraw(base_size = 16)+
      theme(panel.grid.minor = element_blank())+
      scale_size_identity()
  })
  
  
  output$gained_byqtr <- renderPlot({
    ggplot(pbp_grouped_qtr, aes(x=qtr,y=!!as.name('Avg Yds Gained'), group = down))+
      geom_line(aes(size=line_size), color="black")+
      geom_line(aes(size=1.5, color=factor(down)))+
      labs(color="Down")+
      xlab("Quarter")+
      theme_linedraw(base_size = 16)+
      theme(panel.grid.minor = element_blank())+
      scale_size_identity()
  })
  
  
  output$first_gained_lollipop <- renderPlot({
    pbp_grouped_playcall_first <-
      pbp_grouped_playcall %>% 
      filter(down==1)
    
    ggplot(pbp_grouped_playcall_first,aes(x=playcall,y=avg_yds_gained))+
      geom_segment(aes(x=playcall, xend= playcall, y=0, yend=avg_yds_gained), color = "black", lwd = 1.5)+
      geom_point(color="#550000",size=7.5)+
      coord_flip()+
      theme_linedraw(base_size = 16)+
      labs(x=NULL, y="Avg Yds Gained")+
      theme(
            panel.grid.major.y = element_blank(),
            panel.grid.minor.y = element_blank())
    
  })
  
  output$first_conv_lollipop <- renderPlot({
    pbp_grouped_playcall_first <-
      pbp_grouped_playcall %>% 
      filter(down==1)
    
    ggplot(pbp_grouped_playcall_first,aes(x=playcall,y=conversion_pct))+
      geom_segment(aes(x=playcall, xend= playcall, y=0, yend=conversion_pct), color = "black", lwd = 1.5)+
      geom_point(color="#550000",size=7.5)+
      coord_flip()+
      theme_linedraw(base_size = 16)+
      labs(x=NULL, y="Conversion %")+
      theme(
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank())
    
  })
  
  output$second_gained_lollipop <- renderPlot({
    pbp_grouped_playcall_second <-
      pbp_grouped_playcall %>% 
      filter(down==2)
    
    ggplot(pbp_grouped_playcall_second,aes(x=playcall,y=avg_yds_gained))+
      geom_segment(aes(x=playcall, xend= playcall, y=0, yend=avg_yds_gained), color = "black", lwd = 1.5)+
      geom_point(color="#550000",size=7.5)+
      coord_flip()+
      theme_linedraw(base_size = 16)+
      labs(x=NULL, y="Avg Yds Gained")+
      theme(
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank())
    
  })
  
  output$second_conv_lollipop <- renderPlot({
    pbp_grouped_playcall_second <-
      pbp_grouped_playcall %>% 
      filter(down==2)
    
    ggplot(pbp_grouped_playcall_second,aes(x=playcall,y=conversion_pct))+
      geom_segment(aes(x=playcall, xend= playcall, y=0, yend=conversion_pct), color = "black", lwd = 1.5)+
      geom_point(color="#550000",size=7.5)+
      coord_flip()+
      theme_linedraw(base_size = 16)+
      labs(x=NULL, y="Conversion %")+
      theme(
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank())
    
  })
  
  output$third_gained_lollipop <- renderPlot({
    pbp_grouped_playcall_third <-
      pbp_grouped_playcall %>% 
      filter(down==3)
    
    ggplot(pbp_grouped_playcall_third,aes(x=playcall,y=avg_yds_gained))+
      geom_segment(aes(x=playcall, xend= playcall, y=0, yend=avg_yds_gained), color = "black", lwd = 1.5)+
      geom_point(color="#550000",size=7.5)+
      coord_flip()+
      theme_linedraw(base_size = 16)+
      labs(x=NULL, y="Avg Yds Gained")+
      theme(
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank())
    
  })
  
  output$third_conv_lollipop <- renderPlot({
    pbp_grouped_playcall_third <-
      pbp_grouped_playcall %>% 
      filter(down==3)
    
    ggplot(pbp_grouped_playcall_third,aes(x=playcall,y=conversion_pct))+
      geom_segment(aes(x=playcall, xend= playcall, y=0, yend=conversion_pct), color = "black", lwd = 1.5)+
      geom_point(color="#550000",size=7.5)+
      coord_flip()+
      theme_linedraw(base_size = 16)+
      labs(x=NULL, y="Conversion %")+
      theme(
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank())
    
  })
  
  output$fourth_gained_lollipop <- renderPlot({
    pbp_grouped_playcall_fourth <-
      pbp_grouped_playcall %>% 
      filter(down==4)
    
    ggplot(pbp_grouped_playcall_fourth,aes(x=playcall,y=avg_yds_gained))+
      geom_segment(aes(x=playcall, xend= playcall, y=0, yend=avg_yds_gained), color = "black", lwd = 1.5)+
      geom_point(color="#550000",size=7.5)+
      coord_flip()+
      theme_linedraw(base_size = 16)+
      labs(x=NULL, y="Avg Yds Gained")+
      theme(
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank())
    
  })
  
  output$fourth_conv_lollipop <- renderPlot({
    pbp_grouped_playcall_fourth <-
      pbp_grouped_playcall %>% 
      filter(down==4)
    
    ggplot(pbp_grouped_playcall_fourth,aes(x=playcall,y=conversion_pct))+
      geom_segment(aes(x=playcall, xend= playcall, y=0, yend=conversion_pct), color = "black", lwd = 1.5)+
      geom_point(color="#550000",size=7.5)+
      coord_flip()+
      theme_linedraw(base_size = 16)+
      labs(x=NULL, y="Conversion %")+
      theme(
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank())
    
  })
  
  output$run_flow_chart <- renderPlot({
  ggplot(data = pbp_grouped_run_plays,
         aes(axis1 = down, axis2 = playcall, y = avg_yds_gained)) +
    geom_flow(aes(fill = avg_yds_gained)) +
    geom_stratum(fill = "#c5b358", color = "#550000") +
    geom_text(stat = "stratum",
              aes(label = after_stat(stratum))) +
    scale_x_discrete(limits = c("Down", "Playcall"))+
      coord_flip()+
      theme_minimal(base_size = 16)+
    theme(
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank())+
      scale_fill_gradient(low='red',high='darkgreen')
  })
  
####################Playcall tendency
  output$histplot_first <- renderPlot({
    ggplot(first_down_plays,aes(x=playcall))+
      geom_bar(fill="#550000")+
      theme_minimal(base_size = 16)+
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
            panel.grid.major.x = element_blank())
  })
  
  output$histplot_second <- renderPlot({
    ggplot(second_down_plays,aes(x=playcall))+
      geom_bar(fill="#550000")+
      theme_minimal(base_size = 16)+
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
            panel.grid.major.x = element_blank())
  })
  
  output$histplot_third <- renderPlot({
    ggplot(third_down_plays,aes(x=playcall))+
      geom_bar(fill="#550000")+
      theme_minimal(base_size = 16)+
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
            panel.grid.major.x = element_blank())
  })
  
  output$histplot_fourth <- renderPlot({
    ggplot(fourth_down_plays,aes(x=playcall))+
      geom_bar(fill="#550000")+
      theme_minimal(base_size = 16)+
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
            panel.grid.major.x = element_blank())
  })
  
  output$histplot_first_form <- renderPlot({
    ggplot(first_down_plays,aes(x=as.character(Formation)))+
      geom_bar(fill="#550000")+
      xlab("Formation")+
      theme_minimal(base_size = 16)+
      theme( panel.grid.major.x = element_blank())
  })
  
  output$histplot_second_form <- renderPlot({
    ggplot(second_down_plays,aes(x=as.character(Formation)))+
      geom_bar(fill="#550000")+
      xlab("Formation")+
      theme_minimal(base_size = 16)+
      theme( panel.grid.major.x = element_blank())
  })
  
  output$histplot_third_form <- renderPlot({
    ggplot(third_down_plays,aes(x=as.character(Formation)))+
      geom_bar(fill="#550000")+
      xlab("Formation")+
      theme_minimal(base_size = 16)+
      theme( panel.grid.major.x = element_blank())
  })
  
  output$histplot_fourth_form <- renderPlot({
    ggplot(fourth_down_plays,aes(x=as.character(Formation)))+
      geom_bar(fill="#550000")+
      xlab("Formation")+
      theme_minimal(base_size = 16)+
      theme( panel.grid.major.x = element_blank())
  })
  
  output$heatmap_tend <- renderPlot({
  ggplot(pbp_grouped, aes(x = as.character(sort(Formation)), y = playcall, fill = count_)) +
      geom_tile(color = "black", linewidth=.75) +
      geom_text(aes(label = count_), color = "#550000", size =7) +
      scale_fill_gradient(name ="Count:",low = "white", high = "#c5b358") +
      xlab("Formation")+
      ylab("Playcall")+
      theme_minimal(base_size = 20)+
      theme(panel.background = element_rect("gray"),
            panel.grid.major = element_blank(),
            legend.position="bottom",  legend.text = element_text(hjust = 0.5, vjust = 1, angle = 90))
  },width = "auto")
  

      
  
}

shinyApp(ui, server)
