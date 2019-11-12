#!/usr/bin/env Rscript

rm(list = ls(all.names = TRUE))

bookdown::render_book("MovieLens.rmd", output_format = "bookdown::gitbook", output_dir = "books/gitbook")

bookdown::render_book("MovieLens.rmd", output_format = "bookdown::pdf_book", output_dir = "books/pdf")



